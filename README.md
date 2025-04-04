<!--
SPDX-FileCopyrightText: 2025 ProFACE developers

SPDX-License-Identifier: MIT
-->

# ProFACE-Abaqus-Pre

ProFACE-Abaqus-Pre is a python package that provides the `proface.preprocessor.abaqus` plugin for use with the [`proface-pre`](https://github.com/ProFACE-dev/proface-pre) command-line interface (CLI).

This plugin enables the conversion of Abaqus binary `.fil` results files to ProFACE `.h5` FEA input files.

## Installation

Install from <https://pypi.org> with

```
pip install proface-abaqus-pre
```

## Usage

Running

```
proface-pre example.toml
```

will produce `example.h5` from `example.fil`.

`example.toml` has the following format:

```toml
fea_software = "Abaqus"

[Abaqus.input]
fil = "example.fil"  # optional if .toml and .fil have the same name

[Abaqus.results.ref_load]
step = 1  # 0 for last step
increment = 1  # 0 for last increment
```

## License

ProFACE-Abaqus-Pre is licensed under the MIT license.
ProFACE-Pre makes no claims about [ProFACE](https://proface.polimi.it) which is a distinct program with different licensing requirements.

### Disclaimer

This project is not affiliated with, endorsed by, or sponsored by Dassault Systèmes, The HDF Group, or any of their respective products.

- Abaqus is a registered trademark of Dassault Systèmes or its subsidiaries in the United States and/or other countries.
- HDF5 is a trademark of The HDF Group.

All trademarks and registered trademarks are the property of their respective owners.
