# Styler

Ambitious project to render HTML documents with CSS.

## Goals

- Parse CSS, maybe integrate rust-cssparser
  - Tokenizer
  - Parser
  - Media query evaluator
  - Selectors
- Parse HTML to intermediate representation
- CSS Selector matching
- Layout
  - Text sizes -> integrate with HarfBuzz/Platform (Windows: DirectWrite, Mac: CoreText, Linux: ?)
- Render to a surface, PDF?
