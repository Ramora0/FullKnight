using System.Collections;
using System.Collections.Generic;
using WebSocketSharp;
using Newtonsoft.Json;
using UnityEngine;

namespace FullKnight.Net
{
	public class Socket : WebSocket
	{
		public Queue<Message> UnreadMessages { get; private set; } = new Queue<Message>();
		public Message LastMessageSent { get; private set; }

		public Socket(string url, params string[] protocols) : base(url, protocols)
		{
			this.OnMessage += (sender, e) =>
			{
				Message m = JsonConvert.DeserializeObject<Message>(e.Data);
				UnreadMessages.Enqueue(m);
			};
		}

		public void Send(Message data)
		{
			data.sender = "client";
			string textData = JsonConvert.SerializeObject(data);
			base.Send(textData);
			LastMessageSent = data;
		}

		public void SendAsync(Message data, System.Action<bool> completed)
		{
			data.sender = "client";
			string textData = JsonConvert.SerializeObject(data);
			base.SendAsync(textData, completed);
			LastMessageSent = data;
		}

		public class WaitForMessage : CustomYieldInstruction
		{
			private Socket socket;

			public WaitForMessage(Socket socket)
			{
				this.socket = socket;
			}

			public override bool keepWaiting => socket.UnreadMessages.Count == 0;
		}
	}

	public abstract class WebsocketEnv
	{
		public Socket socket;
		protected bool _terminate = false;

		public WebsocketEnv(string url, params string[] protocols)
		{
			socket = new Socket(url, protocols);
		}

		protected void Connect()
		{
			socket.Connect();
		}

		protected abstract IEnumerator Setup();
		protected abstract IEnumerator Dispose();
		protected abstract IEnumerator OnMessage(Message message);

		private IEnumerator _runtime()
		{
			yield return Setup();
			while (true)
			{
				yield return new Socket.WaitForMessage(socket);
				var message = socket.UnreadMessages.Dequeue();
				yield return OnMessage(message);
				if (_terminate) break;
			}
			yield return Dispose();
		}

		protected void SendMessage(Message message)
		{
			message.sender = "client";
			socket.Send(message);
		}

		public void Start()
		{
			GameManager.instance.StartCoroutine(_runtime());
		}

		protected void CloseSocket()
		{
			socket.Close();
		}

		public void Close()
		{
			_terminate = true;
		}
	}
}
