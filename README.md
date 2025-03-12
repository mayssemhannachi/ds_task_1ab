# Project Overview

This project is divided into four main modules, each focusing on a distinct aspect of the system's development. The modules are designed to work together seamlessly, culminating in a comprehensive solution for product recommendation, OCR-based query processing, and image-based product detection.

## Module 1: Data Preparation and Backend Setup

### Task 1: E-commerce Dataset Cleaning

- *Objective*: Ensure the dataset is clean and ready for analysis and vectorization.
- *Key Actions*: Remove duplicates, handle missing values, and standardize formats.

### Task 2: Vector Database Creation

- *Objective*: Set up a vector database using Pinecone to store product vectors.
- *Key Actions*: Define the database schema and integrate with Pinecone.

### Task 3: Similarity Metrics Selection

- *Objective*: Choose and justify the similarity metrics used to compare product vectors.
- *Key Actions*: Evaluate different metrics (e.g., cosine similarity, dot product) and select the best fit based on the dataset characteristics.

### Endpoint 1: Product Recommendation Service

- *Functionality*: Handle natural language queries to recommend products, including safeguards against bad queries and sensitive data exposure.
- *Input*: Customer's natural language query.
- *Output*: Product matches array and a natural language response within specified constraints.

## Module 2: OCR and Web Scraping

### Task 4: OCR Functionality Implementation

- *Objective*: Develop the capability to extract text from images using OCR technology.
- *Key Actions*: Integrate and configure an OCR tool (e.g., Tesseract).

### Task 5: Web Scraping for Product Images

- *Objective*: Scrape product images from e-commerce websites for training data ``CNN_Model_Train_Data.csv``.
- *Key Actions*: Automate scraping, download images, and store them systematically and make sure you have enough data to train the CNN model.

### Endpoint 2: OCR-Based Query Processing

- *Functionality*: Extract and process handwritten queries using the same logic as Endpoint 1.
- *Input*: Image file with handwritten text.
- *Output*: Same output format as Endpoint 1, adapted for image inputs also return the extracted test from OCR.

## Module 3: CNN Model Development

### Task 6: CNN Model Training

- *Objective*: Develop a CNN model from scratch using only the ``products`` mentioned on ``CNN_Model_Train_Data.csv`` to identify products from images.
- *Key Actions*: Train the model using scraped images and clean data without using pre-trained models.

### Endpoint 3: Image-Based Product Detection

- *Functionality*: Use the CNN model to identify products from images and match them using the vector database.
- *Input*: Product image.
- *Output*: Product description and matching products in a format consistent with other endpoints. Also return the name of the `class` that you got from CNN model for the particular input image.

## Module 4: Frontend Development and Integration

### Frontend Page 1: Text Query Interface

- *Features*: Form to submit text queries, display natural language responses, and a product details table.

### Frontend Page 2: Image Query Interface

- *Features*: Allows users to upload images of handwritten queries and displays results similar to Page 1.

### Frontend Page 3: Product Image Upload Interface

- *Features*: Users can upload product images, and view the identified product description and related products in natural language and tabular format.

## Instructions for Presentation

### 1. Incremental Report Writing

Each module completion should be accompanied by a concise, to-the-point report that documents the process, decisions, and outcomes. These reports will be incremental, building upon each other as the bootcamp progresses.

#### Report Format Suggestion:

- *Title Page*: Include the module number and title, the names of the team members, and the submission date.
- *Introduction*: Briefly describe the objectives of the module and its importance to the overall project.
- *High-Level Flow*:
  - *Description*: Outline the main tasks and functionalities developed in the module.
  - *Diagrams*: Include flowcharts or diagrams that visually represent the architecture and data flow.
  - *Key Decisions*: Summarize crucial decisions made during the module, such as choice of technology, design patterns, and configurations.
- *Challenges and Solutions*:
  - Briefly discuss any challenges faced during the module and how they were addressed.
- *Conclusion*: Sum up the outcomes of the module and its readiness for integration with other modules.
- *References*: Cite any tools, libraries, or external resources that were used.

### 2. Video Documentation

Participants are required to create two sets of videos for each module, detailing both the functionality and the technical implementation. This will not only aid in a better understanding of the project but also serve as a reference for future projects.

#### Video Requirements:

- *Functional Demonstration Video*:
  - *Content*: Demonstrate the functionality of each endpoint and page developed in the module.
  - *Focus*: Show how the system responds to various inputs and scenarios. Explain the user interaction with the system.
  - *Duration*: Keep the video concise, preferably under 5 minutes.
- *Code Explanation Video*:
  - *Content*: Provide a high-level overview of the codebase for the module.
  - *Focus*: Explain the structure of the code, major classes, and functions. Highlight any significant patterns or algorithms used.
  - *Duration*: Limit the explanation to under 10 minutes.

### Submission Guidelines:

- *Timing*: Submit the videos along with the incremental report at the end of each module.
- *Format*: Ensure videos are in a common format (e.g., MP4) and quality is sufficient for clear viewing.
- *Hosting*: Upload videos to a platform accessible to all participants and reviewers (e.g., Google Drive, YouTube in unlisted mode). Or you can use loom, fluvid, vmaker etc alternatively.

## Instructions for Coding

### General Guidelines

- *Class-Based Implementation*: It is recommended to use class-based implementation for all backend services to ensure organized, reusable, and maintainable code.
- *Best Practices*:
  - *ACID Properties*: Ensure that database transactions are Atomic, Consistent, Isolated, and Durable to maintain data integrity and reliability.
  - *Modularity*: Build the codebase with clear modularity in mind. Separate different functionalities into distinct modules to enhance readability and maintainability.
- *Packaging*: Organize your code into packages that reflect the services they provide. This approach not only helps in maintaining the code but also simplifies the deployment and scaling process.
- Directories: Whenever you will test on notebook make sure you keep all the notebooks in ``notebook`` directory and use proper naming for the notebooks.

### Tech Stack

- *Web Framework*: Use Flask for developing the backend. Flask provides flexibility and ease of use for setting up API services.
- *Vector Database*: Integrate Pinecone to manage and query vector data efficiently. Pinecone supports scalable vector searches which are crucial for the recommendation systems in this project.
