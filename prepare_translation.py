"""一次性准备本地 Argos 翻译模型；缺少模型时显式联网下载。"""

import os
import shutil
from pathlib import Path

from sam3_service.config import ROOT_DIR


def main():
    os.environ["ARGOS_DEVICE_TYPE"] = "cpu"
    os.environ.setdefault("ARGOS_PACKAGES_DIR", str(ROOT_DIR / "argos-packages"))
    os.environ.setdefault("SAM3_ARGOS_MODEL_DIR", str(ROOT_DIR / "argos_models"))
    os.environ["SAM3_ARGOS_AUTO_INSTALL"] = "0"
    os.environ["SAM3_ARGOS_FORCE_OFFLINE"] = "1"

    from sam3_service import translation

    if not translation.ARGOS_AVAILABLE:
        raise SystemExit("翻译依赖无法导入，请先安装 requirements.txt 并检查上方错误。")

    source, target = translation.ARGOS_SOURCE_CODE, translation.ARGOS_TARGET_CODE
    if translation._pick_installed_package(source, target) is None:
        package = translation.argos_package
        package.update_package_index()
        available = translation._pick_available_package(package.get_available_packages(), source, target)
        if available is None:
            raise SystemExit(f"Argos 官方索引中没有 {source}->{target} 翻译模型。")
        print(f"下载 Argos {source}->{target} 翻译模型...", flush=True)
        downloaded = Path(available.download())
        model_dir = Path(os.environ["SAM3_ARGOS_MODEL_DIR"])
        model_dir.mkdir(parents=True, exist_ok=True)
        local_model = model_dir / downloaded.name
        if downloaded.resolve() != local_model.resolve():
            shutil.copy2(downloaded, local_model)
        package.install_from_path(local_model)

    if source == "zh" and target == "en":
        translated = translation.translate_to_english("人")
        if not translated or translation.contains_chinese(translated):
            raise SystemExit("模型已安装，但中文翻译自检失败，请检查上方错误。")
        print(f"中文翻译自检：人 -> {translated}")
    print(f"Argos {source}->{target} 模型已就绪：{os.environ['ARGOS_PACKAGES_DIR']}")


if __name__ == "__main__":
    main()
