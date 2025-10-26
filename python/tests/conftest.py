import os
import sys
import ctypes


def _maybe_add_built_module_to_path():
    # Locate a locally built tensor_py extension and add its directory to sys.path.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    build_dir = os.path.join(repo_root, 'build')
    search_dirs = []

    if os.path.isdir(build_dir):
        # Prefer active preset via CMAKE_PRESET_NAME if set
        preset = os.environ.get('CMAKE_PRESET_NAME')
        if preset:
            search_dirs.append(os.path.join(build_dir, preset))
        # Shallow scan: build/<preset>
        try:
            for name in os.listdir(build_dir):
                p = os.path.join(build_dir, name)
                if os.path.isdir(p):
                    search_dirs.append(p)
        except OSError:
            pass

    # 1) Check shallow preset dirs
    for d in search_dirs:
        try:
            for f in os.listdir(d):
                if f.startswith('tensor_py.') and (f.endswith('.pyd') or f.endswith('.so')):
                    if os.name == 'nt':
                        try:
                            ctypes.windll.kernel32.SetDllDirectoryW(d)
                        except Exception:
                            pass
                    if d not in sys.path:
                        sys.path.insert(0, d)
                    return
        except OSError:
            continue

    # 2) Fallback: recursive search (handles typical CMake layout build/<preset>/src/python)
    if os.path.isdir(build_dir):
        for root, _dirs, files in os.walk(build_dir):
            for f in files:
                if f.startswith('tensor_py.') and (f.endswith('.pyd') or f.endswith('.so')):
                    d = root
                    if os.name == 'nt':
                        try:
                            ctypes.windll.kernel32.SetDllDirectoryW(d)
                        except Exception:
                            pass
                    if d not in sys.path:
                        sys.path.insert(0, d)
                    return


_maybe_add_built_module_to_path()

def _maybe_add_compiler_bin_to_dll_search_path():
    if os.name != 'nt':
        return
    # Parse CMakeCache.txt to find CMAKE_CXX_COMPILER, then add its directory.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    build_dir = os.path.join(repo_root, 'build')
    cache_candidates = []
    if os.path.isdir(build_dir):
        # Prefer preset env
        preset = os.environ.get('CMAKE_PRESET_NAME')
        if preset:
            cache_candidates.append(os.path.join(build_dir, preset, 'CMakeCache.txt'))
        # Shallow scan
        try:
            for name in os.listdir(build_dir):
                cache_candidates.append(os.path.join(build_dir, name, 'CMakeCache.txt'))
        except OSError:
            pass
    for cache in cache_candidates:
        try:
            with open(cache, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('CMAKE_CXX_COMPILER:'):
                        # format: CMAKE_CXX_COMPILER:FILEPATH=C:/path/to/g++.exe
                        try:
                            compiler_path = line.split('=', 1)[1].strip()
                        except Exception:
                            continue
                        compiler_dir = os.path.dirname(compiler_path)
                        if os.path.isdir(compiler_dir):
                            try:
                                ctypes.windll.kernel32.SetDllDirectoryW(compiler_dir)
                            except Exception:
                                pass
                            os.environ['PATH'] = compiler_dir + os.pathsep + os.environ.get('PATH', '')
                            return
        except OSError:
            continue


_maybe_add_compiler_bin_to_dll_search_path()

