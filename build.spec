import platform
import tomllib

from PyInstaller.building.api import EXE, PYZ
from PyInstaller.building.build_main import Analysis
from PyInstaller.utils.hooks import collect_submodules

with open("pyproject.toml", "rb") as f:
    app_name = tomllib.load(f)["project"]["name"]
build_os = platform.system().lower()
build_arch = platform.machine()

datas = [("src/fahrplangenerator/assets", "assets")]
hiddenimports = collect_submodules("modules")

a_cli = Analysis(
    ["src/fahrplangenerator/main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz_cli = PYZ(a_cli.pure)

exe_cli = EXE(
    pyz_cli,
    a_cli.scripts,
    a_cli.binaries,
    a_cli.datas,
    [],
    name=f"{app_name}-cli-{build_os}-{build_arch}",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

a_api = Analysis(
    ["src/fahrplangenerator/api.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz_api = PYZ(a_api.pure)

exe_api = EXE(
    pyz_api,
    a_api.scripts,
    a_api.binaries,
    a_api.datas,
    [],
    name=f"{app_name}-api-{build_os}-{build_arch}",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
