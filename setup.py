"""
Setup configuration for Smart Forensic Bot
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="smart-forensic-bot",
    version="1.0.0",
    author="Smart Forensic Bot Team",
    author_email="team@smartforensicbot.com",
    description="AI-powered digital forensics analysis system with LangGraph, RAG, and knowledge graph integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saisreekantam/Smart-Forensic-Bot",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "smart-forensic-bot=smart_forensic_bot.cli.main:cli",
        ],
    },
    keywords="forensics, ai, cybersecurity, analysis, investigation, langraph, rag, knowledge-graph",
    project_urls={
        "Bug Reports": "https://github.com/saisreekantam/Smart-Forensic-Bot/issues",
        "Source": "https://github.com/saisreekantam/Smart-Forensic-Bot",
        "Documentation": "https://github.com/saisreekantam/Smart-Forensic-Bot/wiki",
    },
)