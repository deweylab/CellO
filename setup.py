import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="example-pkg-mbernste", # Replace with your own username
    version="1.2.0",
    author="Matthew N. Bernstein",
    author_email="mbernstein@morgridge.org",
    description="Hierarchical cell type classification with the Cell Ontology.",
    long_description="""
    CellO (Cell Ontology-based classification) is a Python package for performing 
    cell type classification of human RNA-seq data. CellO makes hierarchical predictions 
    against the Cell Ontology. These classifiers were trained on most of the human 
    primary cell, bulk RNA-seq data in the Sequence Read Archive.
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/deweylab/CellO",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
