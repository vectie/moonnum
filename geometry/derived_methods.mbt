///|
pub extend Basis3 with @debug.Debug::{to_repr}

///|
pub extend Basis3 with Eq::{not_equal, equal}

///|
pub extend Basis3 with ToJson::{to_json}

///|
pub extend Basis3 with @json.FromJson::{from_json}

///|
pub extend Transform3 with @debug.Debug::{to_repr}

///|
pub extend Transform3 with Eq::{not_equal, equal}

///|
pub extend Transform3 with ToJson::{to_json}

///|
pub extend Transform3 with @json.FromJson::{from_json}

///|
pub extend Vec3 with @debug.Debug::{to_repr}

///|
pub extend Vec3 with Eq::{not_equal, equal}

///|
pub extend Vec3 with ToJson::{to_json}

///|
pub extend Vec3 with @json.FromJson::{from_json}
