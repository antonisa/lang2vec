import setuptools

with open("README.txt", "r") as fh:
    long_description = fh.read()

install_requires = ['numpy',
                    'setuptools',
                    'json']


setuptools.setup(
    name="lang2vec",
    version="0.1.3",
    author="Antonis Anastasopoulos, Patrick Littell, David Mortensen",
    author_email="aanastas@cs.cmu.com",
    description="Returns language vectors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    install_requires=install_requires,
    scripts=['lang2vec/lang2vec.py'],
    packages=['lang2vec'],
    package_dir={'lang2vec': 'lang2vec'},
    package_data={'lang2vec': ['data/*.npz', 'data/*.json']},
    zip_safe=True,
    classifiers=['Operating System :: OS Independent',
               'Programming Language :: Python :: 3',
               'Topic :: Software Development :: Libraries :: Python Modules',
               'Topic :: Text Processing :: Linguistic']
)
