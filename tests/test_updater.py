import hashlib
import io
import os
import subprocess
import sys
import tarfile

import pytest

from transcriptor import updater


def release(tag, *names, **extra):
    return {"tag_name": tag, "assets": [
        {"name": n, "browser_download_url": f"https://x/{n}", "size": 1,
         "digest": "sha256:ab"} for n in names], **extra}


def test_find_update_only_for_newer_published_release(monkeypatch):
    monkeypatch.setattr(updater, "asset_name", lambda: "Transcriptor.exe")
    assert updater.parse_version("v1.10.0") > updater.parse_version("1.9.3")
    assert updater.find_update(release("v1.5.0", "Transcriptor.exe"), "1.4.0")["version"] == "1.5.0"
    assert updater.find_update(release("v1.4.0", "Transcriptor.exe"), "1.4.0") is None
    assert updater.find_update(release("v1.3.0", "Transcriptor.exe"), "1.4.0") is None
    assert updater.find_update(release("v2.0.0", "Transcriptor.exe", prerelease=True), "1.4.0") is None
    assert updater.find_update(release("v2.0.0", "Transcriptor-macOS.dmg"), "1.4.0") is None
    assert updater.find_update(release("nightly", "Transcriptor.exe"), "1.4.0") is None


def served(tmp_path, data, name="asset"):
    """Asset publicado como file:// con su huella real."""
    src = tmp_path / name
    src.write_bytes(data)
    return {"version": "9.0.0", "url": src.as_uri(), "size": len(data),
            "digest": "sha256:" + hashlib.sha256(data).hexdigest()}


def test_download_rejects_wrong_or_missing_digest(tmp_path):
    update = served(tmp_path, b"programa")
    dest = tmp_path / "out"
    seen = []
    updater.download(update, dest, lambda done, total: seen.append((done, total)))
    assert dest.read_bytes() == b"programa" and seen[-1] == (8, 8)
    with pytest.raises(updater.UpdateError, match="dañado"):
        updater.download({**update, "digest": "sha256:00"}, dest, lambda *_: None)
    assert not dest.exists()
    with pytest.raises(updater.UpdateError, match="huella"):
        updater.download({**update, "digest": ""}, dest, lambda *_: None)


def test_install_windows_renames_running_exe(tmp_path, monkeypatch):
    exe = tmp_path / "Transcriptor.exe"
    exe.write_bytes(b"viejo")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(updater, "installed_path", lambda: exe)
    assert updater.install(served(tmp_path, b"nuevo"), lambda *_: None) == exe
    assert exe.read_bytes() == b"nuevo"
    assert (tmp_path / "Transcriptor.old.exe").read_bytes() == b"viejo"
    assert not (tmp_path / ".Transcriptor.exe.new").exists()


def test_install_linux_extracts_executable(tmp_path, monkeypatch):
    target = tmp_path / "bin" / "Transcriptor"
    target.parent.mkdir()
    target.write_bytes(b"viejo")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("Transcriptor")
        info.size = 5
        tar.addfile(info, io.BytesIO(b"nuevo"))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(updater, "installed_path", lambda: target)
    updater.install(served(tmp_path, buf.getvalue()), lambda *_: None)
    assert target.read_bytes() == b"nuevo"
    assert os.name == "nt" or target.stat().st_mode & 0o111  # Windows no tiene permiso de ejecución
    assert sorted(p.name for p in target.parent.iterdir()) == ["Transcriptor"]


def test_install_failure_keeps_current_program(tmp_path, monkeypatch):
    exe = tmp_path / "Transcriptor.exe"
    exe.write_bytes(b"viejo")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(updater, "installed_path", lambda: exe)
    with pytest.raises(updater.UpdateError):
        updater.install({**served(tmp_path, b"nuevo"), "digest": "sha256:00"}, lambda *_: None)
    assert exe.read_bytes() == b"viejo"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Transcriptor.exe", "asset"]


@pytest.mark.skipif(sys.platform != "darwin", reason="usa hdiutil")
def test_install_macos_replaces_app_from_dmg(tmp_path, monkeypatch):
    src = tmp_path / "dmg" / "Transcriptor.app" / "Contents"
    src.mkdir(parents=True)
    (src / "version").write_text("nuevo")
    dmg = tmp_path / "t.dmg"
    folder = str(tmp_path / "dmg")
    # macOS 27 ya no crea imágenes con hdiutil create; antes no existía diskutil image.
    if subprocess.run(["diskutil", "image", "create", "from", folder, "--format", "UDZO",
                       "--volumeName", "T", str(dmg)], capture_output=True).returncode:
        subprocess.run(["hdiutil", "create", "-volname", "T", "-srcfolder", folder,
                        "-format", "UDZO", str(dmg)], check=True, capture_output=True)
    app = tmp_path / "Applications" / "Transcriptor.app"
    (app / "Contents").mkdir(parents=True)
    (app / "Contents" / "version").write_text("viejo")
    monkeypatch.setattr(updater, "installed_path", lambda: app)
    updater.install(served(tmp_path, dmg.read_bytes()), lambda *_: None)
    assert (app / "Contents" / "version").read_text() == "nuevo"
    assert sorted(p.name for p in app.parent.iterdir()) == ["Transcriptor.app"]
