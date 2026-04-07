"""
Manages multiple Hollow Knight instances on Windows.
Creates junction-linked copies of the game directory and spawns/kills processes.
Adapted from HKRL's multi_instance_manager.py.
"""
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
                if fnmatch.fnmatch(name.lower(), "hollow knight.*"):
                    self.exe = os.path.join(self.root, name)
                    break
            break  # only top-level

        if self.exe is None:
            raise FileNotFoundError("Could not find Hollow Knight executable")

        self.instances = []

    def _instance_exe(self, name):
        return self.exe.replace("hollow knight", name).replace("Hollow Knight", name)

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

    def start_instance(self, name):
        if not self._instance_exists(name):
            return False
        try:
            subprocess.Popen(self._instance_exe(name))
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

    def start_all(self):
        for name in self.instances:
            self.start_instance(name)

    def stop_all(self):
        for name in reversed(self.instances):
            self.stop_instance(name)

    def destroy_all(self):
        for name in reversed(list(self.instances)):
            self.stop_instance(name)
            self.delete_instance(name)
