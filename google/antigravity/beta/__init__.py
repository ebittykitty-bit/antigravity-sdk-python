# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Beta namespace and `@beta` decorator for the Google Antigravity SDK.

This package provides:
1. The module namespace (`google.antigravity.beta`) where experimental classes
   and functions decorated with `@beta` are automatically exported and added to
   `__all__`.
2. The `.beta` class/instance namespace (`BetaNamespace`) allowing methods,
   classmethods, staticmethods, and properties decorated with `@beta` on an
   existing class (such as `Agent`) to be accessed via `agent.beta.method()`
   or `Agent.beta.method()` without polluting the class's root namespace.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from typing import Any, Callable, TYPE_CHECKING, TypeVar, overload

_BETA_ATTR = "__antigravity_beta__"
_BETA_MEMBERS_ATTR = "_beta_members"
_CLASS_NS_CACHE_ATTR = "_cached_class_beta_namespace"
_INSTANCE_NS_CACHE_ATTR = "_cached_beta_namespace"

T = TypeVar("T")

_BETA_EXPORTS: dict[str, Any] = {}
_ALL_EXPORTS: list[str] = [
    "BetaNamespace",
    "beta",
    "is_beta",
]

if TYPE_CHECKING:
  __all__: list[str]


def is_beta(obj: Any) -> bool:
  """Returns True if `obj` is marked as a beta API symbol, method, or property."""
  if getattr(obj, _BETA_ATTR, False):
    return True
  if isinstance(obj, property):
    return bool(getattr(obj.fget, _BETA_ATTR, False))
  if isinstance(obj, (classmethod, staticmethod)):
    return bool(getattr(obj.__func__, _BETA_ATTR, False))
  return False


def _mark_beta(obj: Any) -> Any:
  """Marks `obj` (and any underlying descriptor function) with `_BETA_ATTR`."""
  for target in (
      obj,
      obj.fget if isinstance(obj, property) else None,
      obj.__func__ if isinstance(obj, (classmethod, staticmethod)) else None,
  ):
    if target is not None:
      try:
        setattr(target, _BETA_ATTR, True)
      except (AttributeError, TypeError):
        pass
  return obj


def _collect_beta_members(owner: type[Any]) -> dict[str, Any]:
  """Collects all `@beta` members across `owner`'s MRO."""
  merged: dict[str, Any] = {}
  for cls in reversed(owner.__mro__):
    cls_members = cls.__dict__.get(_BETA_MEMBERS_ATTR)
    if cls_members:
      merged.update(cls_members)
  return merged


class _BoundBetaNamespace:
  """Bound `.beta` namespace view for an owning class or instance."""

  def __init__(self, instance: Any, owner: type[Any]) -> None:
    self._instance = instance
    self._owner = owner

  def __getattr__(self, name: str) -> Any:
    if name.startswith("__") and name.endswith("__"):
      raise AttributeError(name)

    for cls in self._owner.__mro__:
      cls_ns = cls.__dict__.get(_CLASS_NS_CACHE_ATTR)
      if cls_ns is not None and name in cls_ns.__dict__:
        return cls_ns.__dict__[name]
      cls_members = cls.__dict__.get(_BETA_MEMBERS_ATTR)
      if cls_members and name in cls_members:
        raw_member = cls_members[name]
        if hasattr(raw_member, "__get__"):
          return raw_member.__get__(self._instance, self._owner)
        return raw_member

    raise AttributeError(
        f"'{self._owner.__name__}.beta' object has no attribute '{name}'"
    )

  def __dir__(self) -> list[str]:
    names = set(_collect_beta_members(self._owner).keys())
    names.update(k for k in self.__dict__ if not k.startswith("_"))
    return sorted(names)

  def __repr__(self) -> str:
    target = (
        f"instance of {self._owner.__name__}"
        if self._instance is not None
        else self._owner.__name__
    )
    return f"<BetaNamespace for {target}>"


class BetaNamespace:
  """Descriptor providing the `.beta` attribute on SDK classes and instances."""

  is_beta = staticmethod(is_beta)

  if TYPE_CHECKING:

    def __getattr__(self, name: str) -> Any:
      ...

  def __call__(self, obj_or_name: Any = None, /) -> Any:
    """Allows `beta` in class scope to act as the `@beta` decorator."""
    return beta(obj_or_name)

  def __get__(self, instance: Any, owner: type[Any] | None = None) -> Any:
    if owner is None:
      owner = type(instance)
    if instance is None:
      cached = owner.__dict__.get(_CLASS_NS_CACHE_ATTR)
      if cached is None:
        cached = _BoundBetaNamespace(None, owner)
        setattr(owner, _CLASS_NS_CACHE_ATTR, cached)
      return cached

    if hasattr(instance, "__dict__"):
      cached = instance.__dict__.get(_INSTANCE_NS_CACHE_ATTR)
      if cached is not None and cached._instance is instance:
        return cached
      cached = _BoundBetaNamespace(instance, owner)
      try:
        object.__setattr__(instance, _INSTANCE_NS_CACHE_ATTR, cached)
      except (AttributeError, TypeError):
        pass
      return cached
    return _BoundBetaNamespace(instance, owner)


class _BetaMethod:
  """Descriptor returned by `@beta` when decorating an owned class member."""

  def __init__(self, fn: Any, custom_name: str | None = None) -> None:
    self.fn = _mark_beta(fn)
    self.custom_name = custom_name

  def __set_name__(self, owner: type[Any], name: str) -> None:
    beta_members = owner.__dict__.get(_BETA_MEMBERS_ATTR)
    if beta_members is None:
      beta_members = {}
      setattr(owner, _BETA_MEMBERS_ATTR, beta_members)
    beta_members[self.custom_name or name] = self.fn
    if owner.__dict__.get(name) is self:
      delattr(owner, name)
    if not any(
        isinstance(base.__dict__.get("beta"), BetaNamespace)
        for base in owner.__mro__
    ):
      setattr(owner, "beta", BetaNamespace())


def _is_class_member(obj: Any) -> bool:
  """Returns True if `obj` is a descriptor or method defined inside a class body."""
  if isinstance(obj, (classmethod, staticmethod, property)):
    return True
  if isinstance(obj, type) or not callable(obj):
    return False
  qualname = getattr(obj, "__qualname__", "")
  parent_scope, sep, _ = qualname.rpartition(".")
  return bool(sep) and not parent_scope.endswith("<locals>")


def _extract_name(obj: Any) -> str:
  """Extracts `__name__` from modules, functions, classes, classmethods, staticmethods, or properties."""
  if isinstance(obj, (classmethod, staticmethod)):
    return getattr(obj.__func__, "__name__", "")
  if isinstance(obj, property) and obj.fget is not None:
    return getattr(obj.fget, "__name__", "")
  name = getattr(obj, "__name__", "")
  if isinstance(obj, type(sys)):
    return name.rpartition(".")[-1]
  return name


@overload
def beta(obj_or_name: None = None, /) -> Callable[[T], T]:
  ...


@overload
def beta(obj_or_name: str, /) -> Callable[[T], T]:
  ...


@overload
def beta(obj_or_name: T, /) -> T:
  ...


def beta(obj_or_name: Any = None, /) -> Any:
  """Exports top-level symbols to `beta` or attaches class methods/properties to `.beta`.

  Args:
    obj_or_name: Either the decorated class/callable/descriptor directly, or an
      optional custom export name string (`@beta("CustomName")`).

  Returns:
    The decorated symbol (for top-level exports) or a `_BetaMethod` descriptor
    (for owned methods/properties inside a class body).
  """

  def _decorate(target: Any, custom_name: str | None = None) -> Any:
    _mark_beta(target)
    if _is_class_member(target):
      return _BetaMethod(target, custom_name=custom_name)
    export_name = custom_name or _extract_name(target)
    if export_name:
      _BETA_EXPORTS[export_name] = target
      globals()[export_name] = target
      if not export_name.startswith("_") and export_name not in _ALL_EXPORTS:
        _ALL_EXPORTS.append(export_name)
        _ALL_EXPORTS.sort()
    return target

  if isinstance(obj_or_name, str):
    return lambda target: _decorate(target, custom_name=obj_or_name)
  if obj_or_name is not None:
    return _decorate(obj_or_name)
  return _decorate


_LAZY_SUBMODULES_LOADED = False
_EXCLUDED_SUBPACKAGES = frozenset({
    "beta",
    "examples",
    "integration_tests",
    "proto",
    "skills",
})


def _ensure_submodules_loaded() -> None:
  """Lazily walks and imports SDK submodules so all `@beta` decorators have executed."""
  global _LAZY_SUBMODULES_LOADED
  if _LAZY_SUBMODULES_LOADED:
    return
  _LAZY_SUBMODULES_LOADED = True
  parent_pkg_name = (__package__ or __name__).rpartition(".")[0]
  if not parent_pkg_name:
    return
  try:
    parent_pkg = importlib.import_module(parent_pkg_name)
  except ImportError:
    return
  pkg_path = getattr(parent_pkg, "__path__", None)
  if not pkg_path:
    return

  def _walk_sdk_modules(paths: Any, prefix: str) -> None:
    for _, name, is_pkg in pkgutil.iter_modules(paths):
      if (
          name.startswith("_")
          or name.endswith("_test")
          or name in _EXCLUDED_SUBPACKAGES
      ):
        continue
      mod_name = f"{prefix}{name}"
      try:
        mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
      except Exception:  # pylint: disable=broad-except
        continue
      if is_pkg:
        sub_path = getattr(mod, "__path__", None)
        if sub_path:
          _walk_sdk_modules(sub_path, f"{mod_name}.")

  _walk_sdk_modules(pkg_path, f"{parent_pkg_name}.")


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
  """PEP 562 module lookup supporting static type checkers and lazy `@beta` discovery."""
  if name == "__all__":
    _ensure_submodules_loaded()
    globals()["__all__"] = _ALL_EXPORTS
    return _ALL_EXPORTS
  if name in _BETA_EXPORTS:
    return _BETA_EXPORTS[name]
  _ensure_submodules_loaded()
  if name in _BETA_EXPORTS:
    return _BETA_EXPORTS[name]
  if name in globals():
    return globals()[name]
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:  # pylint: disable=invalid-name
  """Module `dir()` hook ensuring lazy `@beta` exports are discovered."""
  _ensure_submodules_loaded()
  globals()["__all__"] = _ALL_EXPORTS
  return sorted(set(globals().keys()) | set(_ALL_EXPORTS))
