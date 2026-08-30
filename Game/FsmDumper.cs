using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using HutongGames.PlayMaker;
using UnityEngine;

namespace FullKnight.Game
{
	/// <summary>
	/// Ground-truth extraction of boss behaviour from the live PlayMaker object
	/// graph. Where FsmObserver samples the single string ActiveStateName every
	/// step and leaves Python to *infer* structure from observed transitions,
	/// this reads the structure itself: every state, every transition and the
	/// event that fires it, every action with its serialized parameters, the
	/// FSM's variables, and the animation clip tables.
	///
	/// Nothing here is guessed. HK boss logic is authored as PlayMaker FSMs,
	/// which are serialized data assets; the mod runs inside the game process
	/// with PlayMaker.dll referenced, so the authored graph is directly
	/// readable. What Python currently reconstructs by watching state changes
	/// over thousands of episodes — the transition graph, which states branch,
	/// how long a telegraph lasts, which attacks follow which — is all here
	/// exactly, on the first frame the scene loads:
	///
	///   FsmState.Transitions      -> the real edge set (no observation needed)
	///   SendRandomEvent           -> the real branch distribution (weights[])
	///   Wait.time                 -> the real telegraph duration in seconds
	///   Tk2dPlayAnimation.animName-> the state -> animation clip binding
	///   Fsm.GlobalTransitions     -> interrupt edges (stun/death) that look
	///                                like spurious edges to an observer
	///
	/// Action parameters are walked by reflection rather than by a per-action
	/// switch: HK ships hundreds of custom FsmStateAction subclasses and we
	/// want all of them, including ones we have never heard of. Every public
	/// field is emitted, with PlayMaker's variable wrappers unwrapped to either
	/// a literal value or the name of the variable they read.
	/// </summary>
	public static class FsmDumper
	{
		// Guard against pathological object graphs when walking action fields.
		private const int MaxDepth = 4;
		private const int MaxArray = 64;

		/// <summary>Dump every FSM reachable from the boss subtree plus any FSM
		/// on an active combat collider. `src` tags match FsmObserver's:
		/// B = boss subtree, E = enemy collider, A = knight attack collider.</summary>
		public static Dictionary<string, object> DumpScene(
			string sceneName,
			HashSet<HealthManager> bossHMs,
			HashSet<Collider2D> enemyColliders,
			HashSet<Collider2D> attackColliders)
		{
			var fsms = new List<object>();
			var seen = new HashSet<PlayMakerFSM>();
			var animators = new HashSet<tk2dSpriteAnimator>();

			if (bossHMs != null)
			{
				foreach (var hm in bossHMs)
				{
					if (hm == null || hm.gameObject == null) continue;
					CollectFrom(hm.gameObject, "B", fsms, seen, animators, deep: true);
				}
			}
			CollectFromColliders(enemyColliders, "E", fsms, seen, animators);
			CollectFromColliders(attackColliders, "A", fsms, seen, animators);

			return new Dictionary<string, object>
			{
				["scene"] = sceneName ?? "",
				["schema"] = 1,
				["fsms"] = fsms,
				["animations"] = DumpAnimators(animators),
			};
		}

		private static void CollectFromColliders(
			HashSet<Collider2D> cols, string src, List<object> outFsms,
			HashSet<PlayMakerFSM> seen, HashSet<tk2dSpriteAnimator> animators)
		{
			if (cols == null) return;
			foreach (var col in cols)
			{
				if (col == null || col.gameObject == null) continue;
				CollectFrom(col.gameObject, src, outFsms, seen, animators, deep: false);
			}
		}

		private static void CollectFrom(
			GameObject go, string src, List<object> outFsms,
			HashSet<PlayMakerFSM> seen, HashSet<tk2dSpriteAnimator> animators,
			bool deep)
		{
			// Include inactive children: a boss's later-phase FSMs are often
			// disabled at spawn, and those are exactly the ones an observer
			// never sees until it has already died to them a few hundred times.
			PlayMakerFSM[] found = deep
				? go.GetComponentsInChildren<PlayMakerFSM>(true)
				: go.GetComponents<PlayMakerFSM>();
			if (found != null)
			{
				foreach (var fsm in found)
				{
					if (fsm == null || !seen.Add(fsm)) continue;
					var d = DumpFsm(fsm, src, go.name);
					if (d != null) outFsms.Add(d);
				}
			}

			var anims = deep
				? go.GetComponentsInChildren<tk2dSpriteAnimator>(true)
				: go.GetComponents<tk2dSpriteAnimator>();
			if (anims != null)
				foreach (var a in anims) if (a != null) animators.Add(a);
		}

		// ------------------------------------------------------------ FSM

		private static Dictionary<string, object> DumpFsm(
			PlayMakerFSM pm, string src, string owner)
		{
			try
			{
				var fsm = pm.Fsm;
				if (fsm == null) return null;

				var states = new List<object>();
				var fsmStates = fsm.States;
				if (fsmStates != null)
					foreach (var st in fsmStates)
					{
						var d = DumpState(st);
						if (d != null) states.Add(d);
					}

				return new Dictionary<string, object>
				{
					["src"] = src,
					["owner"] = owner ?? "",
					["path"] = HierarchyPath(pm.gameObject),
					["fsm"] = pm.FsmName ?? "",
					["active"] = pm.isActiveAndEnabled,
					["startState"] = SafeStr(() => fsm.StartState),
					["states"] = states,
					// Any-state transitions. These are the interrupt edges
					// (stun, death, phase change) that an observation-based
					// tracker records as edges out of whatever state happened
					// to be running, fabricating edges that don't exist.
					["globalTransitions"] = DumpGlobalTransitions(fsm),
					["variables"] = DumpVariables(fsm),
				};
			}
			catch (Exception e)
			{
				return new Dictionary<string, object>
				{
					["src"] = src,
					["owner"] = owner ?? "",
					["fsm"] = SafeStr(() => pm.FsmName),
					["error"] = e.GetType().Name + ": " + e.Message,
				};
			}
		}

		private static Dictionary<string, object> DumpState(FsmState st)
		{
			if (st == null) return null;
			var transitions = new List<object>();
			try
			{
				var trs = st.Transitions;
				if (trs != null)
					foreach (var t in trs)
					{
						if (t == null) continue;
						transitions.Add(new Dictionary<string, object>
						{
							["event"] = SafeStr(() => t.EventName),
							["to"] = SafeStr(() => t.ToState),
						});
					}
			}
			catch { }

			var actions = new List<object>();
			try
			{
				// PlayMaker deserializes actions lazily from FsmState.ActionData
				// the first time a state is entered. Touching Actions forces
				// that load, which is what lets us read states the boss has
				// not performed yet in this episode (or ever).
				var acts = st.Actions;
				if (acts != null)
					for (int i = 0; i < acts.Length; i++)
						actions.Add(DumpAction(acts[i], i));
			}
			catch (Exception e)
			{
				actions.Add(new Dictionary<string, object>
				{
					["error"] = "LoadActions: " + e.GetType().Name + ": " + e.Message,
				});
			}

			return new Dictionary<string, object>
			{
				["name"] = SafeStr(() => st.Name),
				["transitions"] = transitions,
				["actions"] = actions,
			};
		}

		private static Dictionary<string, object> DumpAction(FsmStateAction a, int index)
		{
			if (a == null) return new Dictionary<string, object> { ["index"] = index, ["type"] = "null" };
			var d = new Dictionary<string, object>
			{
				["index"] = index,
				["type"] = a.GetType().Name,
			};
			try { d["enabled"] = a.Enabled; } catch { }

			var pars = new Dictionary<string, object>();
			try
			{
				foreach (var f in a.GetType().GetFields(BindingFlags.Public | BindingFlags.Instance))
				{
					// Skip PlayMaker's own bookkeeping fields.
					if (f.Name == "Fsm" || f.Name == "State" || f.Name == "Owner") continue;
					object v;
					try { v = f.GetValue(a); } catch { continue; }
					pars[f.Name] = DumpValue(v, 0);
				}
			}
			catch { }
			d["params"] = pars;
			return d;
		}

		// -------------------------------------------------------- values

		/// <summary>Reduce an arbitrary action field to JSON. PlayMaker wraps
		/// every authored parameter in a NamedVariable: if the author typed a
		/// literal the wrapper's Name is empty and Value holds it; if they
		/// bound a variable, Name is the variable and the value is only
		/// meaningful at runtime. We distinguish the two so downstream can tell
		/// "this wait is always 0.35s" from "this wait reads waitTime".</summary>
		private static object DumpValue(object v, int depth)
		{
			if (v == null) return null;
			if (depth > MaxDepth) return "<depth>";

			if (v is string || v is bool || v is int || v is long
				|| v is float || v is double) return v;
			if (v is Enum) return v.ToString();

			if (v is FsmEvent ev) return new Dictionary<string, object> { ["event"] = ev.Name };

			if (v is NamedVariable nv)
			{
				string name = null;
				try { name = nv.Name; } catch { }
				if (!string.IsNullOrEmpty(name))
					return new Dictionary<string, object> { ["var"] = name };
				return new Dictionary<string, object> { ["value"] = RawValueOf(nv, depth) };
			}

			if (v is Vector2 v2) return new float[] { v2.x, v2.y };
			if (v is Vector3 v3) return new float[] { v3.x, v3.y, v3.z };

			if (v is UnityEngine.Object uo)
				return new Dictionary<string, object>
				{
					["obj"] = uo == null ? null : uo.name,
					["objType"] = v.GetType().Name,
				};

			if (v is IEnumerable en && !(v is string))
			{
				var list = new List<object>();
				int n = 0;
				foreach (var item in en)
				{
					if (n++ >= MaxArray) { list.Add("<truncated>"); break; }
					list.Add(DumpValue(item, depth + 1));
				}
				return list;
			}

			// Composite PlayMaker structs (FsmOwnerDefault, FsmProperty, ...).
			var t = v.GetType();
			if (t.Namespace != null && t.Namespace.StartsWith("HutongGames"))
			{
				var sub = new Dictionary<string, object>();
				foreach (var f in t.GetFields(BindingFlags.Public | BindingFlags.Instance))
				{
					object fv;
					try { fv = f.GetValue(v); } catch { continue; }
					sub[f.Name] = DumpValue(fv, depth + 1);
				}
				if (sub.Count > 0) return sub;
			}

			return v.ToString();
		}

		/// <summary>NamedVariable.RawValue via reflection — the property name is
		/// stable across PlayMaker builds but reading it defensively costs
		/// nothing and keeps a rename from taking the whole dump down.</summary>
		private static object RawValueOf(NamedVariable nv, int depth)
		{
			try
			{
				var p = nv.GetType().GetProperty("RawValue",
					BindingFlags.Public | BindingFlags.Instance);
				if (p != null) return DumpValue(p.GetValue(nv, null), depth + 1);
			}
			catch { }
			try { return nv.ToString(); } catch { return null; }
		}

		private static object DumpGlobalTransitions(Fsm fsm)
		{
			var outList = new List<object>();
			try
			{
				var p = typeof(Fsm).GetProperty("GlobalTransitions",
					BindingFlags.Public | BindingFlags.Instance);
				if (p == null) return outList;
				if (!(p.GetValue(fsm, null) is IEnumerable trs)) return outList;
				foreach (var o in trs)
				{
					if (!(o is FsmTransition t)) continue;
					outList.Add(new Dictionary<string, object>
					{
						["event"] = SafeStr(() => t.EventName),
						["to"] = SafeStr(() => t.ToState),
					});
				}
			}
			catch { }
			return outList;
		}

		private static object DumpVariables(Fsm fsm)
		{
			var d = new Dictionary<string, object>();
			try
			{
				var vars = fsm.Variables;
				if (vars == null) return d;
				// GetAllNamedVariables covers every typed array in one call and
				// keeps working if PlayMaker adds a variable type.
				var m = vars.GetType().GetMethod("GetAllNamedVariables",
					BindingFlags.Public | BindingFlags.Instance);
				if (m == null) return d;
				if (!(m.Invoke(vars, null) is IEnumerable all)) return d;
				foreach (var o in all)
				{
					if (!(o is NamedVariable nv)) continue;
					string n = null;
					try { n = nv.Name; } catch { }
					if (string.IsNullOrEmpty(n)) continue;
					d[n] = new Dictionary<string, object>
					{
						["type"] = nv.GetType().Name,
						["value"] = RawValueOf(nv, 0),
					};
				}
			}
			catch { }
			return d;
		}

		// --------------------------------------------------- animations

		/// <summary>Clip tables off every animator we touched. Gives the exact
		/// frame count / fps / wrap mode per clip, so a state's
		/// Tk2dPlayAnimation action resolves to a real duration in seconds
		/// instead of a name we can only match statistically.</summary>
		private static object DumpAnimators(HashSet<tk2dSpriteAnimator> animators)
		{
			var byLibrary = new Dictionary<string, object>();
			foreach (var a in animators)
			{
				if (a == null) continue;
				try
				{
					var lib = a.Library;
					if (lib == null) continue;
					string key = lib.name ?? a.gameObject.name;
					if (byLibrary.ContainsKey(key)) continue;
					var clips = new List<object>();
					var arr = lib.clips;
					if (arr != null)
						foreach (var c in arr)
						{
							if (c == null) continue;
							int frames = 0;
							try { frames = c.frames != null ? c.frames.Length : 0; } catch { }
							float fps = 0f;
							try { fps = c.fps; } catch { }
							clips.Add(new Dictionary<string, object>
							{
								["name"] = c.name ?? "",
								["frames"] = frames,
								["fps"] = fps,
								["seconds"] = fps > 0f ? frames / fps : 0f,
								["wrapMode"] = SafeStr(() => c.wrapMode.ToString()),
							});
						}
					byLibrary[key] = clips;
				}
				catch { }
			}
			return byLibrary;
		}

		// -------------------------------------------------------- utils

		private static string HierarchyPath(GameObject go)
		{
			if (go == null) return "";
			var parts = new List<string>();
			Transform t = go.transform;
			int depth = 0;
			while (t != null && depth < 12)
			{
				parts.Add(t.gameObject.name);
				t = t.parent;
				depth++;
			}
			parts.Reverse();
			return string.Join("/", parts.ToArray());
		}

		private static string SafeStr(Func<string> f)
		{
			try { return f() ?? ""; } catch { return ""; }
		}
	}
}
