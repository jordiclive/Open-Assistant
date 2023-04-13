from typing import Callable, Optional

from .schemas import ExportMessageNode
from typing import List

def visit_threads_depth_first(
    node: ExportMessageNode,
    visitor: Callable[[List[ExportMessageNode]], None],
    predicate: Optional[Callable[[List[ExportMessageNode]], bool]] = None,
    parents: List[ExportMessageNode] = None,
):
    parents = parents or []
    if not node:
        return
    thread = parents + [node]
    if predicate is None or predicate(thread):
        visitor(thread)
    if node.replies:
        parents = thread
        for c in node.replies:
            visit_threads_depth_first(node=c, visitor=visitor, predicate=predicate, parents=parents)


def visit_messages_depth_first(
    node: ExportMessageNode,
    visitor: Callable[[ExportMessageNode], None],
    predicate: Optional[Callable[[ExportMessageNode], bool]] = None,
):
    if not node:
        return
    if predicate is None or predicate(node):
        visitor(node)
    if node.replies:
        for c in node.replies:
            visit_messages_depth_first(node=c, visitor=visitor, predicate=predicate)
