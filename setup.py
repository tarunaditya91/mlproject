from setuptools import find_packages,setup
from typing import List


ERST="-e ."
def get_requitements(file_path:str)->List[str]:
    
    requirements = []

    with open(file_path) as f:
        requirements=f.readlines()
        requirements = [r.replace("\n","") for r in requirements]

        if ERST in requirements:
            requirements.remove(ERST)
    
    return requirements



setup(
    name='mlproject',
    version='0.0.1',
    description='My Python project',
    author="SHANKER TARUN ADITYA",
    author_email="shankertarunaditya369@gmail.com",  
    packages=find_packages(),
    install_requires=get_requitements('requirements.txt')
    )