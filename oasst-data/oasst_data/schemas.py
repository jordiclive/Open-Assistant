from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from typing import Union, Dict, List

class LabelAvgValue(BaseModel):
    value: Union[float,None]
    count: int


LabelValues = Dict[str, LabelAvgValue]


class ExportMessageEvent(BaseModel):
    type: str
    user_id: Union[str, None]


class ExportMessageEventEmoji(ExportMessageEvent):
    type: Literal["emoji"] = "emoji"
    emoji: str


class ExportMessageEventRating(ExportMessageEvent):
    type: Literal["rating"] = "rating"
    rating: str


class ExportMessageEventRanking(ExportMessageEvent):
    type: Literal["ranking"] = "ranking"
    ranking: List[int]
    ranked_message_ids: List[str]
    ranking_parent_id: Optional[str]
    message_tree_id: Optional[str]
    not_rankable: Optional[bool]  # all options flawed, factually incorrect or unacceptable


class ExportMessageNode(BaseModel):
    message_id: str
    parent_id: Union[str, None]
    user_id: Union[str, None]
    text: str
    role: str
    lang: Union[str, None]
    review_count: Union[str, None]
    review_result: Union[bool, None]
    deleted: Union[bool, None]
    rank: Union[int, None]
    synthetic: Union[bool, None]
    model_name: Union[str, None]
    emojis: Union[Dict[str, int], None]
    replies: Union[List[ExportMessageNode],None]
    labels: Union[LabelValues,None]
    events: Union[Dict[str, List[ExportMessageEvent]],None]


class ExportMessageTree(BaseModel):
    message_tree_id: str
    tree_state: Optional[str]
    prompt: Optional[ExportMessageNode]
    origin: Optional[str]
