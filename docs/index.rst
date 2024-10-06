Welcome to scParadise documentation!
===================================

`scParadise` is a fast, tunable, high-throughput automatic cell type annotation and modality prediction python framework.

`scParadise` includes three sets of tools: 

  1) `scAdam` - fast multi-task multi-class cell type annotation; 
  2) `scEve` - modality prediction; 
  3) `scNoah` - benchmarking cell type annotation and modality prediction. 

`scParadise` enables users to utilize a selection of pre-existing models (`scAdam` or `scEve`) 
as well as to develop and train custom models tailored to specific research needs. 
`scNoah` facilitates the evaluation of novel models and methods for automated cell type annotation 
and modality prediction in scRNA-seq analysis.

scParadise is now in active development. 
If you have any ideas, enhancements, or bug fixes, please feel free to submit a pull request in a [scParadise GitHub repo](https://github.com/Chechekhins/scParadise).


::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Installation {octicon}`plug;1em;`
:link: installation
:link-type: doc

Installation guide for scParadise.
:::

:::{grid-item-card} Tutorials {octicon}`play;1em;`
:link: tutorials/index
:link-type: doc

The tutorials of scParadise models usage in scRNA-seq analysis.
Do you want your cells to be annotated and modalities predicted?
:::

:::{grid-item-card} Models {octicon}`info;1em;`
:link: models/index
:link-type: doc
The list of scAdam and scEve models
:::

:::{grid-item-card} API reference {octicon}`book;1em;`
:link: api/index
:link-type: doc

The API reference of scParadise modules and functions
:::

:::{grid-item-card} GitHub {octicon}`mark-github;1em;`
:link: https://github.com/Chechekhins/scParadise

The repository where you can try to find a solution of your problem
:::
::::

```{toctree}
:hidden: true
:maxdepth: 3
:titlesonly: true

installation
tutorials/index
models/index
api/index
GitHub <https://github.com/Chechekhins/scParadise>
```
