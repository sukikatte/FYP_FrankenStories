# Element Library Collection

This element library was built in two stages:

1. I used `element_library.py` to parse `specification.xml` and produce an initial output file: `element_library_output.json`. During review, I found that some `characters` entries were incomplete (for example, Gemma's lips attribute is not defined in the source "specification" file). I manually refined and normalized the character section into a consistent structure.
4. The final result is `element_library.json`, which is the element library used in this project.
