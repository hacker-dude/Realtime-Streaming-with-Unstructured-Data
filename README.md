
![4c8ca3aa1ede9137bb13e8e9b9f4b5af](https://github.com/hacker-dude/Realtime-Streaming-with-Unstructured-Data/assets/71318439/7bd8b823-c3a1-4d98-a676-6180a5a6f01e)
# Realtime Streaming with Unstructured Data
This project demonstrates how to process and analyze unstructured data in real-time using Apache Spark and AWS S3. It utilizes various data formats, including text, JSON, PDF, CSV, video, and images, to extract valuable insights and information. It provides a foundation for building robust data pipelines for various applications, including job market analysis, sentiment analysis, and customer insights.

## Project Overview
The project consists of the following components:

- Data Sources:
  - Text files
  - JSON files
  - PDF files
  - CSV files
  - Video files
  - Image files
- Data Processing:
  - Spark Streaming for real-time data ingestion
  - User-defined functions (UDFs) for data extraction and cleaning
  - Data transformation and analysis
- Output:
  - Parquet files stored on AWS S3

## Key Features
- __Real-time data processing:__ Enables immediate analysis of incoming data.
- __Unstructured data handling:__ Supports various data formats for comprehensive insights.
- __UDFs for data extraction:__ Custom functions for tailored data processing.
- __AWS S3 integration:__ Stores processed data in a scalable and reliable cloud storage.
Depending on your needs, you might opt for Amazon S3 for durable storage, Amazon DynamoDB for NoSQL database services, or Amazon RDS for relational database services. I’ll opt in for Amazon AWS S3 storage, writing parquet files into our bucket.
- __Crawling, Previewing and Verifying of result:__
The next thing is to use AWS Glue crawler to read the data in the parquets, create a table and store them in a database.


## Project Structure
The project is organized into the following directories:

```
aws-spark-unstructured-streaming
├── config
│ └── config.py # Configuration file for AWS credentials 
├── input # Input folder for different kinds of data types
│ ├── input_text # Directory for text input files
│ ├── input_json # Directory for JSON input files
│ ├── input_pdf # Directory for PDF input files
│ ├── input_csv # Directory for CSV input files
│ ├── input_video # Directory for video input files
│ └── input_image # Directory for image input files
├── docker-compose.yml # Docker Compose file for setting up the environment
├── utils.py # Utility functions for data extraction
├── main.py # Main script containing Spark job
└── README.md # Project documentation
```
## Running the Project

1. Ensure you have the necessary dependencies installed, including Spark, AWS libraries, and Python libraries.
2. Configure the AWS access keys in the ```config.py``` file.
3. Start the containers: ```docker-compose -up -d```
4. Then run: ```docker exec -it aws_spark_unstructured-spark-master-1 spark-submit \
--master spark://spark-master:7077 \
--packages org.apache.hadoop:hadoop-aws:3.3.1,com.amazonaws:aws-java-sdk:1.11.469 \
main.py```
