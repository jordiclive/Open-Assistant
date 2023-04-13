from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class LabelAvgValue(BaseModel):
    value: float
    count: int


LabelValues = dict[str, LabelAvgValue]


class ExportMessageEvent(BaseModel):
    type: str
    user_id: str


class ExportMessageEventEmoji(ExportMessageEvent):
    type: Literal["emoji"] = "emoji"
    emoji: str


class ExportMessageEventRating(ExportMessageEvent):
    type: Literal["rating"] = "rating"
    rating: str


class ExportMessageEventRanking(ExportMessageEvent):
    type: Literal["ranking"] = "ranking"
    ranking: list[int]
    ranked_message_ids: list[str]
    ranking_parent_id: Optional[str]
    message_tree_id: Optional[str]
    not_rankable: Optional[bool]  # all options flawed, factually incorrect or unacceptable


class ExportMessageNode(BaseModel):
    message_id: str
    parent_id: str
    user_id: str
    text: str
    role: str
    lang: str
    review_count: int
    review_result: bool
    deleted: bool
    rank: int
    synthetic: bool
    model_name: str
    emojis: dict[str, int]
    replies: list[ExportMessageNode]
    labels: LabelValues
    events: dict[str, list[ExportMessageEvent]]


class ExportMessageTree(BaseModel):
    message_tree_id: str
    tree_state: Optional[str]
    prompt: Optional[ExportMessageNode]
    origin: Optional[str]
