// javascripts/mathjax.js
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  },
  startup: {
    typeset: false // On laisse MkDocs gérer le rendu
  }
};