"""Chinese prompt normalization and Argos Translate integration."""

import os
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .config import ROOT_DIR

try:
    import argostranslate.package as argos_package
    import argostranslate.translate as argos_translate

    ARGOS_AVAILABLE = True
except Exception:
    argos_package = None
    argos_translate = None
    ARGOS_AVAILABLE = False


translation_available = False
ARGOS_SOURCE_CODE = (os.getenv("SAM3_ARGOS_SOURCE_CODE", "zh") or "zh").strip()
ARGOS_TARGET_CODE = (os.getenv("SAM3_ARGOS_TARGET_CODE", "en") or "en").strip()
ARGOS_TRANSLATOR: Optional[Any] = None
ARGOS_AUTO_INSTALL = os.getenv("SAM3_ARGOS_AUTO_INSTALL", "1").lower() in {"1", "true", "yes", "on"}
ARGOS_MODEL_PATH = (os.getenv("SAM3_ARGOS_MODEL_PATH", "") or "").strip()
ARGOS_MODEL_DIR = (os.getenv("SAM3_ARGOS_MODEL_DIR", "") or "").strip()
ARGOS_FORCE_OFFLINE = os.getenv("SAM3_ARGOS_FORCE_OFFLINE", "0").lower() in {"1", "true", "yes", "on"}
ARGOS_INSTALL_LOCK = threading.Lock()
ARGOS_LOCAL_INSTALL_ATTEMPTED = False
ARGOS_ONLINE_INSTALL_ATTEMPTED = False
ARGOS_TRANSLATION_ERROR_LOGGED = False
ARGOS_OFFLINE_LOCK = threading.Lock()
ARGOS_OFFLINE_TRANSLATOR: Optional[Any] = None
ARGOS_OFFLINE_PACKAGE_PATH: Optional[str] = None
ARGOS_CLEAN_BROKEN_PACKAGES = os.getenv("SAM3_ARGOS_CLEAN_BROKEN_PACKAGES", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _normalize_lang_code(code: str) -> str:
    return (code or "").replace("-", "_").lower()


def _normalize_prompt_label(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    return collapsed.strip(" \t\r\n,;:，；。.!?！？\"'`")


def _lang_matches(code: str, expected_code: str) -> bool:
    normalized = _normalize_lang_code(code)
    expected = _normalize_lang_code(expected_code)
    return normalized == expected or normalized.startswith(f"{expected}_")


def _pick_installed_language(installed_languages: List[Any], expected_code: str) -> Optional[Any]:
    for one_language in installed_languages:
        language_code = getattr(one_language, "code", "")
        if _lang_matches(language_code, expected_code):
            return one_language
    return None


def _pick_available_package(available_packages: List[Any], from_code: str, to_code: str) -> Optional[Any]:
    for one_package in available_packages:
        package_from = getattr(one_package, "from_code", "")
        package_to = getattr(one_package, "to_code", "")
        if _lang_matches(package_from, from_code) and _lang_matches(package_to, to_code):
            return one_package
    return None


def _pick_installed_package(from_code: str, to_code: str) -> Optional[Any]:
    if not ARGOS_AVAILABLE:
        return None
    try:
        installed_packages = argos_package.get_installed_packages()
    except Exception:
        return None
    for one_package in installed_packages:
        package_from = getattr(one_package, "from_code", "")
        package_to = getattr(one_package, "to_code", "")
        if _lang_matches(package_from, from_code) and _lang_matches(package_to, to_code):
            return one_package
    return None


def _discover_argos_package_roots() -> List[Path]:
    roots: List[Path] = []
    env_keys = ["ARGOS_PACKAGE_DIR", "ARGOS_PACKAGES_DIR", "ARGOS_TRANSLATE_PACKAGES_DIR"]
    for one_key in env_keys:
        one_value = (os.getenv(one_key, "") or "").strip()
        if one_value:
            roots.append(Path(one_value))

    roots.append(Path.home() / ".local" / "share" / "argos-translate" / "packages")

    unique_roots: List[Path] = []
    seen: set = set()
    for one_root in roots:
        normalized = str(one_root)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_roots.append(one_root)
    return unique_roots


def _sanitize_argos_package_dirs() -> int:
    if not ARGOS_CLEAN_BROKEN_PACKAGES:
        return 0

    fixed = 0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for root in _discover_argos_package_roots():
        if not root.exists() or not root.is_dir():
            continue
        for one_child in sorted(root.iterdir()):
            if not one_child.is_dir():
                continue
            if ".broken_" in one_child.name:
                continue

            metadata_path = one_child / "metadata.json"
            if metadata_path.exists():
                continue

            quarantine = one_child.with_name(f"{one_child.name}.broken_{ts}")
            suffix_index = 1
            while quarantine.exists():
                quarantine = one_child.with_name(f"{one_child.name}.broken_{ts}_{suffix_index}")
                suffix_index += 1
            try:
                shutil.move(str(one_child), str(quarantine))
                fixed += 1
                print(
                    "Sanitized broken Argos package directory "
                    f"'{one_child}' -> '{quarantine}' (missing metadata.json)"
                )
            except Exception as exc:
                print(f"Failed to sanitize broken Argos package directory '{one_child}': {exc}")
    return fixed


def _configure_stanza_resources_for_offline() -> None:
    # Argos package may include stanza resources under package_path/stanza.
    # Point Stanza there to avoid network downloads in offline environments.
    one_package = _pick_installed_package(ARGOS_SOURCE_CODE, ARGOS_TARGET_CODE)
    if one_package is None:
        return

    stanza_dir = Path(getattr(one_package, "package_path", "")) / "stanza"
    resources_json = stanza_dir / "resources.json"
    if not stanza_dir.exists() or not resources_json.exists():
        return

    resolved = str(stanza_dir)
    os.environ.setdefault("STANZA_RESOURCES_DIR", resolved)
    try:
        import stanza.resources.common as stanza_common  # type: ignore

        stanza_common.DEFAULT_MODEL_DIR = resolved
    except Exception:
        pass

    print(f"Configured STANZA_RESOURCES_DIR for offline use: {resolved}")


def _discover_local_argos_models() -> List[Path]:
    candidates: List[Path] = []

    if ARGOS_MODEL_PATH:
        candidates.append(Path(ARGOS_MODEL_PATH))

    candidate_dirs: List[Path] = []
    if ARGOS_MODEL_DIR:
        candidate_dirs.append(Path(ARGOS_MODEL_DIR))
    candidate_dirs.extend(
        [
            ROOT_DIR / "argos_models",
            ROOT_DIR / "models",
            ROOT_DIR,
        ]
    )

    for one_dir in candidate_dirs:
        if not one_dir.exists() or not one_dir.is_dir():
            continue
        for one_file in sorted(one_dir.glob("*.argosmodel")):
            candidates.append(one_file)

    unique_candidates: List[Path] = []
    seen: set = set()
    for one_path in candidates:
        normalized = str(one_path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(one_path)

    return unique_candidates


def _install_argos_from_local_models() -> bool:
    local_models = _discover_local_argos_models()
    if local_models:
        print(f"Discovered local Argos model files: {[str(one_model) for one_model in local_models]}")
    else:
        print("No local Argos model file discovered (*.argosmodel)")

    installed_any = False
    for model_path in local_models:
        if not model_path.exists() or not model_path.is_file():
            continue
        try:
            print(f"Installing Argos model from local path: {model_path}")
            argos_package.install_from_path(str(model_path))
            print(f"Argos model installed from local path: {model_path}")
            installed_any = True
        except Exception as exc:
            print(f"Failed to install local Argos model {model_path}: {exc}")

    return installed_any


def _load_argos_translator() -> Optional[Any]:
    if not ARGOS_AVAILABLE:
        return None

    try:
        installed_languages = argos_translate.get_installed_languages()
        source_language = _pick_installed_language(installed_languages, ARGOS_SOURCE_CODE)
        target_language = _pick_installed_language(installed_languages, ARGOS_TARGET_CODE)
        if source_language is None or target_language is None:
            return None
        return source_language.get_translation(target_language)
    except Exception as exc:
        print(f"Failed to initialize argostranslate translator: {exc}")
        return None


def _translate_with_argos_model_offline(text: str) -> Optional[str]:
    global ARGOS_OFFLINE_TRANSLATOR, ARGOS_OFFLINE_PACKAGE_PATH

    if not ARGOS_AVAILABLE:
        return None

    with ARGOS_OFFLINE_LOCK:
        one_package = _pick_installed_package(ARGOS_SOURCE_CODE, ARGOS_TARGET_CODE)
        if one_package is None:
            return None

        package_path = str(getattr(one_package, "package_path", ""))
        if not package_path:
            return None

        try:
            import ctranslate2
            from argostranslate import settings as argos_settings
        except Exception:
            return None

        if ARGOS_OFFLINE_TRANSLATOR is None or ARGOS_OFFLINE_PACKAGE_PATH != package_path:
            model_path = str(Path(package_path) / "model")
            ARGOS_OFFLINE_TRANSLATOR = ctranslate2.Translator(
                model_path,
                device=argos_settings.device,
                inter_threads=argos_settings.inter_threads,
                intra_threads=argos_settings.intra_threads,
                compute_type=argos_settings.compute_type,
            )
            ARGOS_OFFLINE_PACKAGE_PATH = package_path

        tokenized = [one_package.tokenizer.encode(text)]
        target_prefix = None
        if getattr(one_package, "target_prefix", ""):
            target_prefix = [[one_package.target_prefix]]

        translated_batches = ARGOS_OFFLINE_TRANSLATOR.translate_batch(
            tokenized,
            target_prefix=target_prefix,
            replace_unknowns=True,
            max_batch_size=1,
            batch_type="tokens",
            beam_size=1,
            num_hypotheses=1,
            return_scores=False,
        )

        if not translated_batches:
            return None

        translated_tokens = translated_batches[0].hypotheses[0]
        value = one_package.tokenizer.decode(translated_tokens)

        if getattr(one_package, "target_prefix", "") and value.startswith(one_package.target_prefix):
            value = value[len(one_package.target_prefix) :]
        if value.startswith(" "):
            value = value[1:]
        return value


def _log_argos_state() -> None:
    if not ARGOS_AVAILABLE:
        return
    try:
        installed_languages = argos_translate.get_installed_languages()
        language_codes = [getattr(one_language, "code", "") for one_language in installed_languages]
        print(f"Argos installed languages: {language_codes}")
    except Exception as exc:
        print(f"Failed to read Argos installed languages: {exc}")

    try:
        installed_packages = argos_package.get_installed_packages()
        package_pairs = [
            f"{getattr(one_package, 'from_code', '')}->{getattr(one_package, 'to_code', '')}"
            for one_package in installed_packages
        ]
        print(f"Argos installed packages: {package_pairs}")
    except Exception as exc:
        print(f"Failed to read Argos installed packages: {exc}")


def _ensure_argos_package_installed() -> bool:
    global ARGOS_LOCAL_INSTALL_ATTEMPTED, ARGOS_ONLINE_INSTALL_ATTEMPTED

    if not ARGOS_AVAILABLE:
        return False

    with ARGOS_INSTALL_LOCK:
        if not ARGOS_LOCAL_INSTALL_ATTEMPTED:
            ARGOS_LOCAL_INSTALL_ATTEMPTED = True
            if _install_argos_from_local_models():
                return True

        if not ARGOS_AUTO_INSTALL:
            return False

        if ARGOS_ONLINE_INSTALL_ATTEMPTED:
            return False
        ARGOS_ONLINE_INSTALL_ATTEMPTED = True

        try:
            print(f"Attempting to auto-install Argos package {ARGOS_SOURCE_CODE}->{ARGOS_TARGET_CODE} from network ...")
            argos_package.update_package_index()
            available_packages = argos_package.get_available_packages()
            package_to_install = _pick_available_package(
                available_packages,
                from_code=ARGOS_SOURCE_CODE,
                to_code=ARGOS_TARGET_CODE,
            )
            if package_to_install is None:
                print(f"No Argos package found for {ARGOS_SOURCE_CODE}->{ARGOS_TARGET_CODE}")
                return False

            package_path = package_to_install.download()
            argos_package.install_from_path(package_path)
            print(f"Argos package {ARGOS_SOURCE_CODE}->{ARGOS_TARGET_CODE} installed from network")
            return True
        except Exception as exc:
            print(f"Failed to auto-install Argos package {ARGOS_SOURCE_CODE}->{ARGOS_TARGET_CODE} from network: {exc}")
            return False


if ARGOS_AVAILABLE:
    _sanitize_argos_package_dirs()
    _log_argos_state()
    _configure_stanza_resources_for_offline()
    ARGOS_TRANSLATOR = _load_argos_translator()
    translation_available = ARGOS_TRANSLATOR is not None
    if translation_available:
        print(f"Translation package {ARGOS_SOURCE_CODE}->{ARGOS_TARGET_CODE} is ready")
    else:
        if _ensure_argos_package_installed():
            _log_argos_state()
            ARGOS_TRANSLATOR = _load_argos_translator()
            translation_available = ARGOS_TRANSLATOR is not None

    if translation_available:
        print(f"Translation package {ARGOS_SOURCE_CODE}->{ARGOS_TARGET_CODE} is ready")
    else:
        print(
            f"Translation package {ARGOS_SOURCE_CODE}->{ARGOS_TARGET_CODE} not ready at startup; "
            "will retry during requests"
        )
        if ARGOS_AUTO_INSTALL:
            print("Argos auto-install is enabled")
        if ARGOS_MODEL_PATH:
            print(f"SAM3_ARGOS_MODEL_PATH configured: {ARGOS_MODEL_PATH}")
        if ARGOS_MODEL_DIR:
            print(f"SAM3_ARGOS_MODEL_DIR configured: {ARGOS_MODEL_DIR}")
    if ARGOS_FORCE_OFFLINE:
        print("SAM3_ARGOS_FORCE_OFFLINE is enabled")
else:
    print("argostranslate is not installed, Chinese prompt translation disabled")


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]+", text or ""))


def translate_to_english(text: str) -> str:
    normalized_text = _normalize_prompt_label(text)
    if not contains_chinese(normalized_text):
        return normalized_text or text

    if not ARGOS_AVAILABLE:
        return normalized_text

    global ARGOS_TRANSLATOR, translation_available, ARGOS_TRANSLATION_ERROR_LOGGED

    try:
        if ARGOS_FORCE_OFFLINE:
            translated = _translate_with_argos_model_offline(normalized_text)
            if not translated:
                return normalized_text
        else:
            if ARGOS_TRANSLATOR is None:
                # Lazy retry avoids a startup-time false negative permanently disabling translation.
                ARGOS_TRANSLATOR = _load_argos_translator()

            if ARGOS_TRANSLATOR is None and _ensure_argos_package_installed():
                ARGOS_TRANSLATOR = _load_argos_translator()

            if ARGOS_TRANSLATOR is not None:
                translation_available = True
                translated = ARGOS_TRANSLATOR.translate(normalized_text)
            else:
                translated = argos_translate.translate(normalized_text, ARGOS_SOURCE_CODE, ARGOS_TARGET_CODE)

        translated = _normalize_prompt_label(translated).lower()
        if not translated or contains_chinese(translated):
            return normalized_text
        return translated
    except Exception as exc:
        translated_offline = _translate_with_argos_model_offline(normalized_text)
        if translated_offline:
            translated = _normalize_prompt_label(translated_offline).lower()
            if translated and not contains_chinese(translated):
                return translated

        if not ARGOS_TRANSLATION_ERROR_LOGGED:
            print(f"argostranslate failed for '{normalized_text}', using raw text: {exc}")
            ARGOS_TRANSLATION_ERROR_LOGGED = True
        return normalized_text


def split_prompt_classes(prompt_text: str) -> List[str]:
    if not prompt_text:
        return []
    return [item.strip() for item in re.split(r"[;；,，]+", prompt_text) if item.strip()]


def prepare_single_text_prompt(prompt_text: Optional[str]) -> Tuple[Optional[List[str]], Optional[str], Optional[str], bool]:
    """Prepare one SAM3 text prompt. SAM3 box+text grounding uses a single prompt id."""
    original_prompt = _normalize_prompt_label(prompt_text or "")
    if not original_prompt:
        return None, None, None, False

    translated_prompt = translate_to_english(original_prompt)
    translated_prompt = _normalize_prompt_label(translated_prompt).lower() if translated_prompt else original_prompt
    if not translated_prompt:
        translated_prompt = original_prompt

    return [translated_prompt], original_prompt, translated_prompt, translated_prompt != original_prompt
