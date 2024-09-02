---
layout: default
parent: CB&B 634 Computational Methods in Informatics
grand_parent: Courses at Yale
title: "Good enough practices in scientific computing"
nav_order: 1
discuss: true
math: katex
---

# "Good Enough Practices in Scientific Computing"

[Paper Link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510)

## Introduction

Computational skills are essential for modern scientific research, but many researchers aren't formally trained in these areas. This often results in data mismanagement, inefficient workflows, and irreproducible studies. The paper "Good Enough Practices in Scientific Computing" provides actionable guidelines for researchers to enhance their data management, coding practices, collaboration, and project organization, making their research more efficient and reproducible. This blog post dives into these practices, providing detailed steps, examples, and key points to help you improve your research workflow.

## Why "Good Enough" Practices Matter

"Good enough" practices are not about perfection—they are about practicality and efficiency. These practices are designed to be straightforward and effective, helping researchers improve their computational workflows without requiring deep technical expertise. By adopting these guidelines, you can make incremental improvements that have a significant impact on the quality and reproducibility of your research.

## Core Practices for Better Scientific Computing

### 1. Data Management

Effective data management ensures your research is reproducible and your data is secure and accessible.

- **Save Raw Data**: Never overwrite raw data; always keep an unchanged copy to ensure you can redo analyses from scratch if needed. Use file permissions to prevent accidental modifications.

  > **Example**: Protect raw data by setting it to read-only.
  ```bash
  cp survey_data_original.csv survey_data.csv
  chmod 444 survey_data_original.csv  # Read-only
  ```

- **Back Up Data in Multiple Locations**: Use multiple storage solutions (cloud services, external drives, institutional servers) to back up your data. This redundancy protects against data loss from hardware failures or accidental deletions.

  > **Example**: Back up data to AWS S3.
  ```bash
  aws s3 cp survey_data.csv s3://myresearchbackup/
  ```

- **Create Tidy Data**: Organize your data in a "tidy" format where each variable is a column, each observation is a row, and each type of observational unit is stored in a separate table. This standardization simplifies data analysis and sharing.

  > **Example**: Ensure each column represents a variable, and each row represents an observation.  
  ```plaintext
  | ID | Age | Gender | Treatment | Outcome |
  |----|-----|--------|-----------|---------|
  | 1  | 45  | Male   | A         | 5.4     |
  | 2  | 50  | Female | B         | 6.1     |
  ```

- **Document All Data Processing Steps**: Use scripts or notebooks to automate and document data processing steps. This documentation ensures your process is transparent and reproducible.

  > **Example**: Document data cleaning steps in a Python script.

  ```python
  # data_cleaning.py
  import pandas as pd
  
  # Load raw data
  df = pd.read_csv('raw_data.csv')
  
  # Clean data by removing missing values
  df_clean = df.dropna().reset_index(drop=True)
  
  # Save cleaned data
  df_clean.to_csv('clean_data.csv', index=False)
  ```

**Key Points for Data Management:**

- Store data in open formats like CSV, JSON, or HDF5 for long-term accessibility.
- Use unique identifiers for each data record to facilitate merging and linking datasets.
- Submit data to repositories with DOIs to make it citable and accessible to others.

### 2. Writing and Organizing Code

Good coding practices improve code readability, maintainability, and reproducibility.

- **Write Modular Code**: Break your scripts into small, reusable functions that each perform a single task. This modular approach makes your code easier to understand, test, and debug.

  > **Example**: Refactor code into functions.

  ```py
  def load_data(file_path):
      """Load data from a CSV file."""
      return pd.read_csv(file_path)

  def clean_data(df):
      """Clean data by dropping missing values."""
      return df.dropna().reset_index(drop=True)

  # Main execution
  df = load_data('raw_data.csv')
  df_clean = clean_data(df)
  df_clean.to_csv('clean_data.csv', index=False)
  ```

- **Use Meaningful Names**: Choose descriptive names for variables, functions, and scripts. Avoid generic names like `x`, `y`, or `temp`.

  > **Example**: Use `survey_data` or `cleaned_data` instead of `df`.

- **Include Documentation**: Write comments to explain what each part of your code does, and include a brief description at the start of each script or function.

  > **Example**: Adding comments to code.

  ```py
  # Load raw survey data
  df = pd.read_csv('raw_data.csv')
  
  # Remove missing values and reset index
  df_clean = df.dropna().reset_index(drop=True)
  ```

**Key Points for Software Development:**

- Use version control (e.g., Git) for tracking changes and collaborative development.
- Avoid hard-coding values; use configuration files or command-line arguments instead.
- Regularly refactor code to reduce complexity and eliminate duplication.
- Write automated tests to ensure your code works as expected.

### 3. Collaboration

Effective collaboration involves clear communication, shared goals, and structured project management.

- **Create a README File**: Every project should have a `README.md` file that explains the project's purpose, setup instructions, and contribution guidelines. This file is often the first resource new collaborators consult.

  > **Example**: Basic structure of a `README.md` file.

  ```md
  # Project: Survey Data Analysis
  
  This project analyzes survey data to identify trends in public opinion.
  
  ## Setup Instructions
  Install required Python packages:

  > pip install -r requirements.txt
  
  ## Contributing
  Please see the `CONTRIBUTING.md` file for guidelines.
  ```

- **Use Version Control**: Implement version control with Git to manage changes and facilitate collaboration. Platforms like GitHub and GitLab provide tools for issue tracking, code review, and pull requests.

  > **Example Git Commands**:
  ```bash
  git init
  git add .
  git commit -m "Initial commit with data and scripts"
  git push origin main
  ```

- **Maintain a Shared To-Do List**: Use a shared to-do list or issue tracker to manage tasks and track progress. This could be a simple text file (`todo.md`) or a list of issues on GitHub.

  > **Example**: GitHub Issues for task management.
  ```md
  - [ ] Clean raw data
  - [ ] Perform initial data analysis
  - [ ] Write introduction for manuscript
  ```

**Key Points for Collaboration:**

- Agree on a code style guide and enforce it using linters (e.g., Pylint for Python).
- Conduct regular code reviews to improve code quality and knowledge sharing.
- Use collaborative tools like Google Docs for writing or Overleaf for LaTeX projects.
- Clearly define roles and responsibilities within the team.

### 4. Project Organization

A well-organized project structure makes it easier to manage and navigate files, understand workflows, and collaborate.

- **Adopt a Consistent Directory Structure**: Organize your files into a logical directory structure that separates raw data, processed data, scripts, results, and documentation.

  > **Example Directory Structure**:
  ```
  project/
  ├── data/
  │   ├── raw/
  │   └── processed/
  ├── scripts/
  ├── results/
  └── docs/
  ```

- **Use Clear and Descriptive File Names**: File names should clearly reflect their contents or function. Avoid generic names like `file1.csv` or `data.csv`; instead, use descriptive names like `2024-01-20_experiment-results.csv`.

  > **Example**:  
  Use `survey_data_2024.csv` instead of `data.csv`.

**Key Points for Project Organization:**

- Store all project-related files in a single top-level directory.
- Separate code, data, and results into distinct subdirectories.
- Include a `LICENSE` file specifying the terms under which others can use your work.
- Maintain a `CHANGELOG.md` to track major changes and updates to the project.

### 5. Tracking Changes

Tracking changes effectively helps maintain the integrity of your project and facilitates collaboration.

- **Manual Versioning**: If you are not using a version control system, manually save different versions of your files with version numbers or dates in the file names.

  > **Example**: Manual versioning of data files.
  ```
  survey_data_v1_2024-01-20.csv
  survey_data_v2_2024-01-21.csv
  ```

- **Automated Version Control**: Use Git for automatic version control. This is particularly useful for collaborative projects and helps prevent loss of work.

  > **Example Git Commands**:
  ```bash
  git commit -am "Refactored data cleaning script"
  git push
  ```

**Key Points for Tracking Changes:**

- Use branches in Git for developing new features or conducting experiments.
- Merge branches frequently to avoid large, complex merges.
- Document significant changes in commit messages and the `CHANGELOG.md` file.

### 6. Writing Manuscripts

Efficiently managing the writing process is essential for successful scientific communication.

- **Use Collaborative Writing Tools**: Google Docs is excellent for real-time collaboration, allowing multiple authors to edit and comment simultaneously. For more control, consider using Markdown or LaTeX with Git to version control manuscripts.

  > **Example Markdown Template for a Manuscript**:

  ```md
  # Title of the Paper
  
  ## Abstract
  Brief summary of the research.
  
  ## Introduction
  Background and motivation for the study.
  
  ## Methods
  Detailed description of the methods used.
  ```

- **Version-Controlled Manuscripts**: Write your manuscript in plain text using Markdown or LaTeX and manage it with Git for full version control and collaboration features.

  > **Example**: Use a Git repository on GitHub to manage manuscript drafts and revisions.

**Key Points for Writing Manuscripts:**

- Use reference management tools like Zotero or Mendeley for handling citations.
- Maintain a consistent writing style across all sections.
- Regularly update the manuscript based on feedback and new data.
- Ensure that all supplementary materials are well-documented and accessible.

## Conclusion

Implementing these "good enough" practices can significantly enhance the efficiency, reproducibility, and collaborative potential of your scientific research. These practices provide a strong foundation for building more sophisticated workflows as your skills develop. Start with these basics, and continually refine your processes over time. Remember, the goal is progress, not perfection. By integrating these guidelines into your daily workflow, you'll make your research more robust, transparent, and valuable to the scientific community.

By adopting these practices, you'll ensure your research is organized, reproducible, and ready for future challenges. Make these practices a part of your daily routine, and you'll contribute to a more robust and reliable scientific enterprise.