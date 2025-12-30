from setuptools import setup, find_packages

def get_requirements(file_path) -> list[str]:
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    return [req.strip() for req in requirements if req.strip() and not req.startswith(('#', '-e'))]

setup(
    name='Cancer-Diagnosis-MLOps-Pipeline',
    version='0.0.1',
    author='Kelvin Omoruyi',
    author_email='kelvin.omoruyi.dev@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)