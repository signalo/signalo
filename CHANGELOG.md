# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please make sure to add your changes to the appropriate categories:

- `Added`: for new functionality
- `Changed`: for changes in existing functionality
- `Deprecated`: for soon-to-be removed functionality
- `Removed`: for removed functionality
- `Fixed`: for fixed bugs
- `Performance`: for performance-relevant changes
- `Security`: for security-relevant changes
- `Other`: for everything else

## [Unreleased]

### Added

- Added sliding window filters for moving minimum/maximum/bounds.

### Changed

- Replaced own `CircularBuffer` with `circular_buffer` crate.
- Updated dependencies:
  - `dimensioned` from `0.7` to `0.8`
  - `guts` from `0.1.1` to `0.2.0`

### Deprecated

- n/a

### Removed

- n/a

### Fixed

- Fixed `Mean` filter incorrectly doubling the first input value (#126).
- Fixed variance calculation in `MeanVariance` filter.

### Performance

- Reduced redundant clones in Kalman filter.
- Used `take()` pattern in `MeanVariance` sink to reduce clones.

### Security

- n/a

### Other

- n/a

## [0.6.0] - 2021-06-28

See commit log.

## [0.5.0] - 2018-10-19

See commit log.

## [0.4.0] - 2018-10-07

See commit log.

## [0.3.1] - 2018-09-27

See commit log.

## [0.3.0] - 2018-09-26

See commit log.

## [0.2.0] - 2018-08-28

See commit log.

## [0.1.5] - 2018-08-02

See commit log.

## [0.1.3] - 2018-06-26

See commit log.

## [0.1.2] - 2018-05-18

See commit log.

## [0.1.1] - 2018-05-11

See commit log.

## [0.1.0] - 2018-05-02

Initial release.
