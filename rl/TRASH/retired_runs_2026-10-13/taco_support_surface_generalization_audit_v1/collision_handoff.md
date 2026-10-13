# Collision handoff (read-only)

1. Official SPIDER XHand disables broad contacts by default.
2. Runtime collision uses explicit collision pairs.
3. The official scene generator self-collision set is not all-pairs.
4. The project scene builder retains the pinned 30 intrahand pairs.
5. Historical Pour evidence preserves the same 30/20/82 taxonomy.
6. Omitted shell overlap is not automatically a runtime collision failure.
7. Native-material real self-collision remains a separate audit question.
8. The 82 omitted pairs must not be added wholesale to the solver.

Collision code/topology diff in this audit: 0.
