from pyspark.sql import SparkSession
from config.config import configuration as cfg
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
from pyspark.sql.functions import udf, regexp_replace
from utils import *


# Define UDFs (User Defined Functions) for data extraction
def define_udfs():
    return {
        "extract_file_name": udf(extract_file_name, StringType()),
        "extract_position": udf(extract_position, StringType()),
        "extract_requirments": udf(extract_req, StringType()),
        "extract_notes": udf(extract_notes, StringType()),
        "extract_education_length": udf(extract_education_length, StringType()),
        "extract_experience_length": udf(extract_experience_length, StringType()),
        "extract_selection": udf(extract_selection, StringType()),
        "extract_duties": udf(extract_duties, StringType()),
        "extract_salary": udf(
            extract_salary,
            StringType(
                [
                    StructField("salary_start", DoubleType(), True),
                    StructField("salary_end", DoubleType(), True),
                ]
            ),
        ),
        "extract_classcode": udf(extract_class_code, StringType()),
        "extract_start_date": udf(extract_start_date, DateType()),
        "extract_end_date": udf(extract_end_date, DateType()),
        "extract_application_location": udf(extract_application_location, StringType()),
    }


def main():
    # Initialize Spark session with AWS S3 configurations
    spark = (
        SparkSession.builder.appName("AWS_Spark_Unstructured")
        .config(
            "spark.jars.packaged",
            "org.apache.hadoop:hadoop-aws:3.3.1" "com.amazonaws:aws-kava-sdk:1.11.469",
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", cfg.get("AWS_ACCESS_KEY"))
        .config("spark.hadoop.fs.s3a.secret.key", cfg.get("AWS_SECRET_KEY"))
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .getOrCreate()
    )

    # Define input directories for various file formats
    text_input_dir = "input\input_text"
    json_input_dir = "input\input_json"
    pdf_input_dir = "input\input_pdf"
    csv_input_dir = "input\input_csv"
    video_input_dir = "input\input_video"
    image_ipur_dir = "input\input_image"

    # Define schema for JSON data
    data_schema = StructType(
        [
            StructField("file_name", StringType(), True),
            StructField("position", StringType(), True),
            StructField("classcode", StringType(), True),
            StructField("salary_start", DoubleType(), True),
            StructField("salary_end", DoubleType(), True),
            StructField("start_date", DateType(), True),
            StructField("end_date", DateType(), True),
            StructField("req", StringType(), True),
            StructField("notes", StringType(), True),
            StructField("duties", StringType(), True),
            StructField("selection", StringType(), True),
            StructField("experience_length", StringType(), True),
            StructField("job_type", StringType(), True),
            StructField("education_length", StringType(), True),
            StructField("school_type", StringType(), True),
            StructField("application_location", StringType(), True),
        ]
    )

    # Retrieve UDFs
    udfs = define_udfs()

    # Read streaming data from text files
    job_bulletins_df = (
        spark.readStrem.format("text").option("wholetext", "true").load(text_input_dir)
    )

    # Read streaming data from JSON files
    json_df = spark.readStrem.json(json_input_dir, schema=data_schema, multiline=True)

    # Clean and process text data using UDFs
    job_bulletins_df = job_bulletins_df.withColumn(
        "file_name", regexp_replace(udf["extract_file_name"]("value"), r"\r", " ")
    )
    job_bulletins_df = job_bulletins_df.withColumn(
        "value", regexp_replace(regexp_replace("value", r"\n", " "), r"\r", " ")
    )
    job_bulletins_df = job_bulletins_df.withColumn(
        "position", regexp_replace(udf["extract_position"]("value"), r"\r", " ")
    )
    job_bulletins_df = job_bulletins_df.withColumn(
        "salary_start", udfs["salary_start"]("value").getField("salary_start")
    )
    job_bulletins_df = job_bulletins_df.withColumn(
        "salary_end", udfs["salary_end"]("value").getField("salary_end")
    )
    job_bulletins_df = job_bulletins_df.withColumn(
        "start_date", udfs["extract_start_date"]("value")
    )
    job_bulletins_df = job_bulletins_df.withColumn(
        "end_date", udfs["extract_end_date"]("value")
    )
    job_bulletins_df = job_bulletins_df.withColumn(
        "classcode", udfs["extract_classcode"]("value")
    )
    job_bulletins_df = job_bulletins_df.withColumn(
        "req", udfs["extract_requirments"]("value")
    )
    job_bulletins_df = job_bulletins_df.withColumn(
        "notes", udfs["extract_notes"]("value")
    )
    job_bulletins_df = job_bulletins_df.withColumn(
        "duties", udfs["extract_duties"]("value")
    )
    job_bulletins_df = job_bulletins_df.withColumn(
        "selection", udfs["extract_selection"]("value")
    )
    job_bulletins_df = job_bulletins_df.withColumn(
        "experience_length", udfs["extract_experience_length"]("value")
    )
    job_bulletins_df = job_bulletins_df.withColumn(
        "education_length", udfs["extract_education_length"]("value")
    )
    job_bulletins_df = job_bulletins_df.withColumn(
        "application_location", udfs["extract_application_location"]("value")
    )

    # Select relevant columns from processed text data
    job_bulletins_df = job_bulletins_df.select(
        "file_name",
        "position",
        "start_date",
        "end_date",
        "salary_start",
        "salary_end",
        "classcode",
        "req",
        "notes",
        "duties",
        "selection",
        "experience_length",
        "education_length",
        "application_location",
    )

    # Select relevant columns from JSON data
    json_df = json_df.select(
        "file_name",
        "position",
        "start_date",
        "end_date",
        "salary_start",
        "salary_end",
        "classcode",
        "req",
        "notes",
        "duties",
        "selection",
        "experience_length",
        "education_length",
        "application_location",
    )

    # Union the dataframes from text and JSON sources
    union_df = job_bulletins_df.union(json_df)

    # Define a function to write the stream to a specified output
    def streamWriter(input, checkpointFolder, output):
        return (
            input.writeStream.outputMode("append")
            .format("parquet")
            .option("checkpointLocation", checkpointFolder)
            .option("path", output)
            .trigger(processingTime="5 seconds")
            .start()
        )

    # Write the unioned dataframe to the console for debugging
    query = (
        union_df.writeStream.outputMode("append")
        .format("console")
        .option("truncate", False)
        .start()
    )

    # Write the unioned dataframe to AWS S3
    query = streamWriter(
        union_df,
        "s3a://spark-unstructured-streaming/checkpoints/",
        "s3a://spark-unstructured-streaming/data/spark_unstructured",
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
