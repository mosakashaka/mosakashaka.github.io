# ztree_toc_boot

Generate sidebar TOC for markdown content.
Based on [i5ting_ztree_toc](https://github.com/i5ting/i5ting_ztree_toc).

Basically, one needs to copy and past the markdowns to a template file and I feel like it still a little hard.
So I wrote lines of code to generate a template based on the already-generated markdown html.
Paste below reference `js` and `css` files at the begenning of the markdown html to get the sidebar TOC work.

```html
    <script type="text/javascript" src="jquery-1.4.4.min.js"></script>
    <script type="text/javascript" src="jquery.ztree.core-3.5.js"></script>
    <script type="text/javascript" src="ztree_toc.js"></script>
    <script type="text/javascript" src="ztree_toc_boot.js"></script>
    <link rel="stylesheet" href="zTreeStyle.css" type="text/css">
    <link rel="stylesheet" href="ztree_toc_boot.css" type="text/css">
```

It's still a little work but seems easier to me though....