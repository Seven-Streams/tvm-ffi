# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Python ``import`` modelling for the ``tvm-ffi-stubgen`` Python backend.

Import representation is language-specific: Python uses ``from mod import name
[as alias]`` with module-prefix rewrites (:data:`.consts.MOD_MAP`), whereas
other languages (e.g. Rust ``use a::b::c``) follow entirely different rules.
This module therefore owns the Python representation; the language-agnostic
pipeline only handles an opaque collector (:class:`PythonImports`).
"""

from __future__ import annotations

import dataclasses

from . import consts as C


@dataclasses.dataclass(frozen=True, eq=True)
class ImportItem:
    """An import statement item."""

    mod: str
    name: str
    type_checking_only: bool = False
    alias: str | None = None

    def __init__(
        self,
        name: str,
        type_checking_only: bool = False,
        alias: str | None = None,
    ) -> None:
        """Initialize an `ImportItem` with the given module name and optional alias."""
        if "." in name:
            mod, name = name.rsplit(".", 1)
            for mod_prefix, mod_replacement in C.MOD_MAP.items():
                if mod.startswith(mod_prefix):
                    mod = mod.replace(mod_prefix, mod_replacement, 1)
                    break
        else:
            mod = ""
        object.__setattr__(self, "mod", mod)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "type_checking_only", type_checking_only)
        object.__setattr__(self, "alias", alias)

    @property
    def name_with_alias(self) -> str:
        """Generate a string of the form `name as alias` if an alias is set, otherwise just `name`."""
        return f"{self.name} as {self.alias}" if self.alias else self.name

    @property
    def full_name(self) -> str:
        """Generate a string of the form `mod.name` or `name` if no module is set."""
        return f"{self.mod}.{self.name}" if self.mod else self.name

    def __repr__(self) -> str:
        """Generate an import statement string for this item."""
        return str(self)

    def __str__(self) -> str:
        """Generate an import statement string for this item."""
        if self.mod:
            ret = f"from {self.mod} import {self.name_with_alias}"
        else:
            ret = f"import {self.name_with_alias}"
        return ret


@dataclasses.dataclass
class PythonImports:
    """Opaque import collector threaded through the Python generation pipeline.

    The language-agnostic ``cli`` treats this as an opaque token: it asks the
    backend to create one, seed it from ``import-object`` directives, and later
    render it. Only the Python backend reaches inside.
    """

    items: list[ImportItem] = dataclasses.field(default_factory=list)
    has_lib_load: bool = False
    """Whether an FFI library-loading import was seen (adds ``LIB`` to ``__all__``)."""
