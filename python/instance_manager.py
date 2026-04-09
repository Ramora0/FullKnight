"""
Manages multiple Hollow Knight instances on Windows.
Creates junction-linked copies of the game directory and spawns/kills processes.
Adapted from HKRL's multi_instance_manager.py.
"""
import atexit
import os
import fnmatch
import shutil
import subprocess
import psutil


class InstanceManager:
    def __init__(self, hollow_knight_dir, data_dir="Hollow Knight_Data"):
        self.root = hollow_knight_dir
        self.data_dir = data_dir

        if not os.path.exists(self.root):
            raise FileNotFoundError(f"HK root not found: {self.root}")
        if not os.path.exists(os.path.join(self.root, self.data_dir)):
            raise FileNotFoundError(f"HK data dir not found: {os.path.join(self.root, self.data_dir)}")

        self.steam_app_id = os.path.join(self.root, "steam_appid.txt")

        self.exe = None
        for _, _, files in os.walk(self.root):
            for name in files:
                if fnmatch.fnmatch(name.lower(), "hollow?knight.*"):
                    self.exe = os.path.join(self.root, name)
                    break
            break  # only top-level

        if self.exe is None:
            raise FileNotFoundError("Could not find Hollow Knight executable")

        self.instances = []

        # Locate Steam API DLL so we can disable it during multi-instance runs
        self._steam_api = os.path.join(self.root, self.data_dir, "Plugins", "x86_64", "steam_api64.dll")
        self._steam_api_bak = self._steam_api + ".bak"

    def _instance_exe(self, name):
        exe_basename = os.path.basename(self.exe)
        return os.path.join(self.root, exe_basename.replace(os.path.splitext(exe_basename)[0], name))

    def _instance_data(self, name):
        return os.path.join(self.root, name + "_Data")

    def _instance_exists(self, name):
        return os.path.exists(self._instance_data(name)) or os.path.exists(self._instance_exe(name))

    def create_instance(self, name):
        if self._instance_exists(name):
            if name not in self.instances:
                self.instances.append(name)
            return True

        try:
            # Create junction link for data directory
            subprocess.check_call(
                'mklink /J "%s" "%s"' % (
                    self._instance_data(name),
                    os.path.join(self.root, self.data_dir)
                ),
                shell=True
            )
            # Write steam app ID
            with open(self.steam_app_id, "w") as f:
                f.write("367520")

            # Copy executable
            shutil.copyfile(self.exe, self._instance_exe(name))
            self.instances.append(name)
            return True
        except Exception as e:
            print(f"Failed to create instance {name}: {e}")
            return False

    def delete_instance(self, name):
        if not self._instance_exists(name):
            return False
        try:
            exe_path = self._instance_exe(name)
            data_path = self._instance_data(name)
            if os.path.exists(exe_path):
                os.remove(exe_path)
            if os.path.exists(data_path):
                os.rmdir(data_path)  # rmdir works for junctions
            if name in self.instances:
                self.instances.remove(name)
            return True
        except Exception as e:
            print(f"Failed to delete instance {name}: {e}")
            return False

    def start_instance(self, name, graphical=False):
        if not self._instance_exists(name):
            return False
        try:
            if graphical:
                cmd = [self._instance_exe(name)]
            else:
                cmd = [
                    self._instance_exe(name),
                    "-batchmode",
                    "-screen-width", "64",
                    "-screen-height", "64",
                    "-screen-quality", "0",
                    "-screen-fullscreen", "0",
                    "-nolog",
                ]
            subprocess.Popen(cmd)
            return True
        except Exception as e:
            print(f"Failed to start instance {name}: {e}")
            return False

    def stop_instance(self, name):
        try:
            for proc in psutil.process_iter(["name"]):
                if proc.info["name"] == name + ".exe":
                    proc.terminate()
                    proc.wait(timeout=10)
            return True
        except Exception as e:
            print(f"Error stopping instance {name}: {e}")
            return False

    def spawn_n(self, n):
        """Create N instances named i0, i1, ..., i(n-1)."""
        for i in range(n):
            self.create_instance(f"i{i}")

    def _disable_steam_api(self):
        """Rename steam_api64.dll to prevent Steam from auto-launching."""
        if os.path.exists(self._steam_api):
            os.rename(self._steam_api, self._steam_api_bak)
            atexit.register(self._restore_steam_api)

    def _restore_steam_api(self):
        """Restore steam_api64.dll."""
        if os.path.exists(self._steam_api_bak):
            os.rename(self._steam_api_bak, self._steam_api)

    def start_all(self, graphical=False):
        self._disable_steam_api()
        for name in self.instances:
            self.start_instance(name, graphical=graphical)

    def stop_all(self):
        for name in reversed(self.instances):
            self.stop_instance(name)
        self._restore_steam_api()

    def destroy_all(self):
        for name in reversed(list(self.instances)):
            self.stop_instance(name)
            self.delete_instance(name)
