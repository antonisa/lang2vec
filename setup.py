import setuptools
import urllib.request, sys

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ['numpy',
                    'setuptools']

def query_yes_no(question, default="yes"):
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

question = "Do you want to download pre-computed distances?\nThey require about 240MB in your disk. Proceed? "
if query_yes_no(question):
    try:
        print("Downloading pre-computed distances... (will take a few seconds).")
        filename, headers = urllib.request.urlretrieve("http://www.cs.cmu.edu/~aanastas/files/distances.zip", "lang2vec/data/distances.zip")
    except:
        raise Exception("Failed to download the distances :(")


setuptools.setup(
    name="lang2vec",
    version="1.1.4",
    author="Patrick Littell, David Mortensen, Antonis Anastasopoulos",
    author_email="aanastas@cs.cmu.com",
    description="A simple library for querying the URIEL typological database.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/antonisa/lang2vec",
    install_requires=install_requires,
    scripts=['lang2vec/lang2vec.py'],
    packages=['lang2vec'],
    package_dir={'lang2vec': 'lang2vec'},
    package_data={'lang2vec': ['data/*.npz', 'data/*.json', 'data/learned.npy', 'data/distances.zip', 'data/distances_languages.txt']},
    zip_safe=True,
    classifiers=['Operating System :: OS Independent',
               'Programming Language :: Python :: 3',
               'Topic :: Software Development :: Libraries :: Python Modules',
               'Topic :: Text Processing :: Linguistic']
)
