from dataclasses import dataclass
from typing import List, Optional, Any, Dict, TypeVar, Callable, Type, cast


T = TypeVar("T")


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except AssertionError:
            pass
    assert False


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_str_to_float(x: Any) -> float:
    assert isinstance(x, str)
    return float(x)


def from_str_to_int(x: Any) -> int | None:
    assert isinstance(x, str)
    if x == "":
        return None
    return int(x)


def from_empty_str(x: Any) -> str | None:
    assert isinstance(x, str)
    if x == "":
        return None
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def to_float(x: Any) -> float:
    assert isinstance(x, (int, float))
    return x


def from_dict(f: Callable[[Any], T], x: Any) -> Dict[str, T]:
    assert isinstance(x, dict)
    return {k: f(v) for (k, v) in x.items()}


@dataclass
class HierarchyStop:
    stop_id: str
    stop_name: str
    stop_lat: float
    stop_lon: float
    zone_id: Optional[str] = None
    location_type: Optional[int] = None
    parent_station: Optional[str] = None
    level_id: Optional[str] = None
    children: Optional[List["HierarchyStop"]] = None

    @staticmethod
    def from_dict(obj: Any) -> "HierarchyStop":
        assert isinstance(obj, dict)
        stop_id = from_str(obj.get("stop_id"))
        stop_name = from_str(obj.get("stop_name"))
        stop_lat = from_str_to_float(obj.get("stop_lat"))
        stop_lon = from_str_to_float(obj.get("stop_lon"))
        zone_id = from_union([from_empty_str, from_none], obj.get("zone_id"))
        location_type = from_union([from_str_to_int, from_none], obj.get("location_type"))
        parent_station = from_union([from_empty_str, from_none], obj.get("parent_station"))
        level_id = from_union([from_empty_str, from_none], obj.get("level_id"))
        children = from_union([lambda x: from_list(HierarchyStop.from_dict, x), from_none], obj.get("children"))
        return HierarchyStop(stop_id, stop_name, stop_lat, stop_lon, zone_id, location_type, parent_station, level_id, children)

    def to_dict(self) -> dict:
        result: dict = {}
        result["stop_id"] = from_str(self.stop_id)
        result["stop_name"] = from_str(self.stop_name)
        result["stop_lat"] = to_float(self.stop_lat)
        result["stop_lon"] = to_float(self.stop_lon)
        result["zone_id"] = from_union([from_none, from_str], self.zone_id)
        result["location_type"] = from_union([from_int, from_none], self.location_type)
        result["parent_station"] = from_union([from_none, from_str], self.parent_station)
        result["level_id"] = from_union([from_none, from_str], self.level_id)
        if self.children is not None:
            result["children"] = from_union([lambda x: from_list(lambda x: to_class(HierarchyStop, x), x), from_none], self.children)
        return result


def hierarchy_stop_from_dict(s: Any) -> Dict[str, HierarchyStop]:
    return from_dict(HierarchyStop.from_dict, s)


def hierarchy_stop_to_dict(x: Dict[str, HierarchyStop]) -> Any:
    return from_dict(lambda x: to_class(HierarchyStop, x), x)
