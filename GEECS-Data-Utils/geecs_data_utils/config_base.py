"""
Shared utilities for managing configuration directories and cached lookups.

Provides a small helper class that handles:
- Validating and setting a base configuration directory
- Clearing and accessing a simple path cache
- Finding config files by name using configurable filename patterns
- Bootstrapping from an environment variable and optional fallback resolver
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

PathLike = Union[str, Path]


class ConfigDirManager:
    """
    Manage a base configuration directory and cached config path lookups.

    Parameters
    ----------
    env_var : str
        Environment variable to bootstrap the base directory from (if set).
    logger : logging.Logger
        Logger used for info/warning messages.
    name : str, optional
        Human-friendly name used in log messages.
    fallback_resolver : Callable[[], Optional[Path]], optional
        Optional callable that returns a default base directory if no env var
        is set. Only called when the environment variable is unset.
    fallback_name : str, optional
        Label for the fallback source used in log messages.
    """

    def __init__(
        self,
        *,
        env_var: str,
        logger: logging.Logger,
        name: str = "Config",
        fallback_resolver: Optional[Callable[[], Optional[Path]]] = None,
        fallback_name: Optional[str] = None,
    ) -> None:
        self.env_var = env_var
        self.logger = logger
        self.name = name
        self.fallback_resolver = fallback_resolver
        self.fallback_name = fallback_name or "fallback resolver"

        self._base_dir: Optional[Path] = None
        self._cache: dict[str, Path] = {}

    @property
    def base_dir(self) -> Optional[Path]:
        """Return the currently configured base directory."""
        return self._base_dir

    @property
    def cache(self) -> dict[str, Path]:
        """Expose the internal cache (useful for external inspection)."""
        return self._cache

    def set_base_dir(self, path: PathLike) -> Path:
        """
        Validate and set the base configuration directory.

        Raises
        ------
        FileNotFoundError
            If the specified path does not exist.
        NotADirectoryError
            If the specified path is not a directory.
        """
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Config base dir does not exist: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"Config base dir is not a directory: {p}")

        self._base_dir = p
        self.clear_cache(log=False)
        self.logger.info("%s base dir set to %s", self.name, p)
        return p

    def clear_cache(self, *, log: bool = True) -> None:
        """Clear the cached config paths."""
        self._cache.clear()
        if log:
            self.logger.info("%s cache cleared", self.name)

    def bootstrap_from_env_or_fallback(self) -> None:
        """
        Initialize the base directory using env var or optional fallback resolver.

        If the environment variable is set but invalid, a warning is logged and
        no fallback is attempted (mirroring existing behavior).
        """
        env_dir = os.getenv(self.env_var)
        env_ok = False
        if env_dir:
            try:
                self.set_base_dir(env_dir)
                self.logger.info(
                    "Loaded %s dir from %s: %s", self.name, self.env_var, env_dir
                )
                env_ok = True
            except Exception as exc:  # pragma: no cover - log only
                self.logger.warning("%s invalid: %s", self.env_var, exc)

        if env_ok:
            return

        if self.fallback_resolver:
            try:
                if fallback_dir := self.fallback_resolver():
                    self.set_base_dir(fallback_dir)
                    self.logger.info(
                        "Loaded %s dir from %s: %s",
                        self.name,
                        self.fallback_name,
                        fallback_dir,
                    )
            except Exception as exc:  # pragma: no cover - log only
                self.logger.warning("%s invalid: %s", self.fallback_name, exc)

    def find_config(
        self,
        name: str,
        *,
        patterns: Sequence[str],
        config_dir: Optional[Path] = None,
        use_cache: bool = True,
        missing_base_message: str,
        not_found_label: str = "Config",
    ) -> Path:
        """
        Resolve a config path by name using configurable filename patterns.

        Parameters
        ----------
        name : str
            Logical config name (file stem).
        patterns : Sequence[str]
            Filename patterns that may include ``{name}`` placeholders.
        config_dir : Optional[Path], default=None
            Explicit base directory to search. Falls back to the stored base dir.
        use_cache : bool, default=True
            Whether to use and update the cached results.
        missing_base_message : str
            Error message used when no base directory is set.
        not_found_label : str, default="Config"
            Label used when constructing the FileNotFoundError message.

        Returns
        -------
        Path
            Resolved config file path.
        """
        base = config_dir or self._base_dir
        if base is None:
            raise ValueError(missing_base_message)

        cache_key = f"{base}::{name}"
        if use_cache and cache_key in self._cache:
            cached_path = self._cache[cache_key]
            if cached_path.exists():
                return cached_path
            self._cache.pop(cache_key, None)

        rendered_patterns = [pat.format(name=name) for pat in patterns]

        for pat in rendered_patterns:
            p = base / pat
            if p.exists():
                self.logger.info("Found config (direct): %s", p)
                self._cache[cache_key] = p
                return p

        all_matches = []
        for pat in rendered_patterns:
            all_matches.extend(base.rglob(pat))

        if not all_matches:
            raise FileNotFoundError(
                f"{not_found_label} '{name}' not found under {base}\n"
                "Searched recursively for patterns:\n"
                + "\n".join(f"  - {pat}" for pat in rendered_patterns)
            )

        all_matches.sort()
        if len(all_matches) > 1:
            self.logger.warning(
                f"Multiple configs found for '{name}':\n"
                + "\n".join(f"  - {match}" for match in all_matches)
                + f"\nUsing: {all_matches[0]}"
            )

        result = all_matches[0]
        self.logger.info("Found config (recursive): %s", result)
        self._cache[cache_key] = result
        return result
