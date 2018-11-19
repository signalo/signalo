![jumbotron](./jumbotron.png)

# signalo

## Synopsis

A DSP toolbox with focus on embedded environments.

## About

Signalo basically consists of four basic [traits](traits) and implementations thereof:

- [`Source<T>`](sources): `() -> T`
- [`Filter<T>`](filters): `T -> U`
- [`Sink<T>`](sinks): `T -> ()`
- `Finalize`: `() -> U`

Roughly signalo's traits are equivalent in semantics to the following stdlib APIs:

- `Source<…>` ≈ `core::iter::Iterator<…>`
- `Filter<…>` ≈ `core::iter::Map<…>`
- `Sink<…>` & `Finalize` ≈ `Iterator::fold(…)`
- `Filter<…>` & `Finalize` ≈ `core::iter::Scan<…>`

Types implementing `Finalize` usually also implement either `Filter<T>` or `Sink<T>`.

Signalo provides the **basic building-blocks** for **low-level real-time filtering pipelines**,  
which can be **assembled via composition** either manually or through the use of [pipes](pipes).

## Workspace

-  [signalo](signalo): [![Build Status](http://img.shields.io/travis/signalo/signalo.svg?style=flat-square)](https://travis-ci.org/signalo/signalo)
[![Downloads](https://img.shields.io/crates/d/signalo.svg?style=flat-square)](https://crates.io/crates/signalo/)
[![Version](https://img.shields.io/crates/v/signalo.svg?style=flat-square)](https://crates.io/crates/signalo/)
[![License](https://img.shields.io/crates/l/signalo.svg?style=flat-square)](https://crates.io/crates/signalo/)
- [signalo_traits](traits): [![Build Status](http://img.shields.io/travis/signalo/signalo.svg?style=flat-square)](https://travis-ci.org/signalo/signalo)
[![Downloads](https://img.shields.io/crates/d/signalo_traits.svg?style=flat-square)](https://crates.io/crates/signalo_traits/)
[![Version](https://img.shields.io/crates/v/signalo_traits.svg?style=flat-square)](https://crates.io/crates/signalo_traits/)
[![License](https://img.shields.io/crates/l/signalo_traits.svg?style=flat-square)](https://crates.io/crates/signalo_traits/)
- [signalo_filters](filters): [![Build Status](http://img.shields.io/travis/signalo/signalo.svg?style=flat-square)](https://travis-ci.org/signalo/signalo)
[![Downloads](https://img.shields.io/crates/d/signalo_filters.svg?style=flat-square)](https://crates.io/crates/signalo_filters/)
[![Version](https://img.shields.io/crates/v/signalo_filters.svg?style=flat-square)](https://crates.io/crates/signalo_filters/)
[![License](https://img.shields.io/crates/l/signalo_filters.svg?style=flat-square)](https://crates.io/crates/signalo_filters/)
- [signalo_sinks](sinks): [![Build Status](http://img.shields.io/travis/signalo/signalo.svg?style=flat-square)](https://travis-ci.org/signalo/signalo)
[![Downloads](https://img.shields.io/crates/d/signalo_sinks.svg?style=flat-square)](https://crates.io/crates/signalo_sinks/)
[![Version](https://img.shields.io/crates/v/signalo_sinks.svg?style=flat-square)](https://crates.io/crates/signalo_sinks/)
[![License](https://img.shields.io/crates/l/signalo_sinks.svg?style=flat-square)](https://crates.io/crates/signalo_sinks/)
- [signalo_sources](sources): [![Build Status](http://img.shields.io/travis/signalo/signalo.svg?style=flat-square)](https://travis-ci.org/signalo/signalo)
[![Downloads](https://img.shields.io/crates/d/signalo_sources.svg?style=flat-square)](https://crates.io/crates/signalo_sources/)
[![Version](https://img.shields.io/crates/v/signalo_sources.svg?style=flat-square)](https://crates.io/crates/signalo_sources/)
[![License](https://img.shields.io/crates/l/signalo_sources.svg?style=flat-square)](https://crates.io/crates/signalo_sources/)
- [signalo_pipes](pipes): [![Build Status](http://img.shields.io/travis/signalo/signalo.svg?style=flat-square)](https://travis-ci.org/signalo/signalo)
[![Downloads](https://img.shields.io/crates/d/signalo_pipes.svg?style=flat-square)](https://crates.io/crates/signalo_pipes/)
[![Version](https://img.shields.io/crates/v/signalo_pipes.svg?style=flat-square)](https://crates.io/crates/signalo_pipes/)
[![License](https://img.shields.io/crates/l/signalo_pipes.svg?style=flat-square)](https://crates.io/crates/signalo_pipes/)

![](dependencies.png)

## Contributing

Please read [CONTRIBUTING.md](../CONTRIBUTING.md) for details on our [code of conduct](https://www.rust-lang.org/conduct.html),
and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/signalo/signalo/tags).

## Authors

* **Vincent Esche** – *Initial work* – [Regexident](https://github.com/Regexident)

See also the list of [contributors](https://github.com/signalo/signalo/contributors) who participated in this project.

## License

This project is licensed under the [**MPL-2.0**](https://www.tldrlegal.com/l/mpl-2.0) – see the [LICENSE.md](LICENSE.md) file for details.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you to the licensor shall be under the terms and conditions of this license, without any additional terms or conditions. Notwithstanding the above, nothing herein shall supersede or modify the terms of any separate license agreement you may have executed with licensor regarding such contributions.
