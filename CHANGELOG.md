# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- re-worked bytecode assembly and added tests
- the use of sockets to transmit object data instead of saving it in a code
  object. As a result, there is no memory overhead related to serialized data
  while morph `pyc` files remain very small
- added experimentation with AWS EC2

### Fixed

- cell variables

## [0.2.1] - 2022-02-21

### Added

- `fork_shell` to fork process
- Multiple process spawn

### Fixed

- Switched to a modern build system

## [0.2] - 2022-01-31

### Added

+ A new stack method based on bytecode prediction
+ Support for custom save/load routines
+ Initial support for frame stack continuity checking

### Fixed

+ Generator states

### Removed

+ CPython 3.8 support

## [0.1.2] - 2021-09-14

### Added

+ `except`, `with` clauses
+ REPL mode

### Fixed

+ Missing and redundant items on stack
+ Stack items scopes

## [0.1.1] - 2021-08-28

### Added

+ Initial `for` and `try`
+ 3.10 compatibility

### Fixed

+ Reorganized the code for python version compatibility

## [0.1] - 2021-07-29

### Added

+ Serialize generators

### Fixed

+ Cleanup source code
+ Pure-python morphs (no longer needs memory hacks)

## [0.0] - 2021-07-13

### Added

+ Proof of concept

