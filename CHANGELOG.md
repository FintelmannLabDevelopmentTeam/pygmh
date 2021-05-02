# Changelog

## [unreleased]

## 0.3.1

- Fix: Corrected package version string in pyproject.toml to match set tag

## 0.3.0

- Chg: Now logging in level `DEBUG` from persistence adapter
- Add: Added parameter `overwrite_if_existing` to `pygmh.persistence.IAdapter#write()`

## 0.2.0

- Chg: Refactored individual cli commands into single `app.py`
- Add: Alias for `pygmh.persistence.gmh.Adapter` as `pygmh.persistence.GmhAdapter` and `pygmh.persistence.interface.IAdapter` as `pygmh.persistence.Adapter` 
- Add: Added build status badge to README.md

## 0.1.0

- Initial implementation
