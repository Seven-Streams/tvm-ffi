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
"""Compatibility facade for Python stub codegen.

Kept temporarily while code moves into ``tvm_ffi.stub.python_backend``.
"""

from __future__ import annotations

from .python_backend.codegen import (
    RenderType,
    _type_suffix_and_record,
    render_func_signature,
    render_object_ffi_init,
    render_object_fields,
    render_object_init,
    render_object_methods,
)
from .python_backend.codegen import generate_python_all as generate_all
from .python_backend.codegen import generate_python_export as generate_export
from .python_backend.codegen import generate_python_ffi_api as generate_ffi_api
from .python_backend.codegen import generate_python_global_funcs as generate_global_funcs
from .python_backend.codegen import generate_python_import_section as generate_import_section
from .python_backend.codegen import generate_python_init as generate_init
from .python_backend.codegen import generate_python_object as generate_object

__all__ = [
    "RenderType",
    "_type_suffix_and_record",
    "generate_all",
    "generate_export",
    "generate_ffi_api",
    "generate_global_funcs",
    "generate_import_section",
    "generate_init",
    "generate_object",
    "render_func_signature",
    "render_object_ffi_init",
    "render_object_fields",
    "render_object_init",
    "render_object_methods",
]
