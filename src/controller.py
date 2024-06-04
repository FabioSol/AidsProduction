import pandas as pd
import psycopg2

HOST = "localhost"
PORT = 5432
DATABASE = "aids"
USER = "aids"
PASSWORD = "aids"

class Controller:
    def __init__(self):
        self.conn = None  # Initialize connection object
        self.connect_to_db()

    def connect_to_db(self):
        self.conn = psycopg2.connect(
            host=HOST, port=PORT, database=DATABASE, user=USER, password=PASSWORD
        )

    def validate_cols(self, df, table_schema):
        """
        Validates if DataFrame columns match expected schema for a table.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            table_schema (dict): A dictionary containing expected column names and types.

        Returns:
            bool: True if columns match, False otherwise.
        """
        expected_cols = set(table_schema.keys())
        df_cols = set(df.columns)

        if expected_cols != df_cols:
            missing_cols = expected_cols - df_cols
            extra_cols = df_cols - expected_cols
            error_msg = ""
            if missing_cols:
                error_msg += f"Missing columns: {', '.join(missing_cols)}\n"
            if extra_cols:
                error_msg += f"Extra columns: {', '.join(extra_cols)}"
            raise ValueError(error_msg)
        return True

    def insert_data(self, df, table_name):
        """Inserts data from DataFrame into a PostgreSQL table.

        Args:
            df (pd.DataFrame): The DataFrame containing data to insert.
            table_name (str): The name of the target table in the database.
        """
        self.connect_to_db()  # Ensure connection is established

        if not self.conn:
            raise ConnectionError("Failed to connect to database.")

        try:
            cursor = self.conn.cursor()
            # Assuming you know the column names and their data types
            columns = ", ".join(df.columns)
            placeholders = ", ".join(["%s"] * len(df.columns))
            sql = f"""INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"""
            for index, row in df.iterrows():
                cursor.execute(sql, tuple(row.values))
            self.conn.commit()
            print(f"Data inserted into {table_name} table.")
        except Exception as e:
            self.conn.rollback()  # Rollback on errors
            raise e
        finally:
            if self.conn:
                cursor.close()

    def insert_new_data(self, df):
        """Inserts new data from DataFrame into the 'new_data' table.

                Args:
                    df (pd.DataFrame): The DataFrame containing new data to insert.
                """

        self.insert_data(df, "new_data")

    def insert_predictions(self, df):
        """Inserts predictions from DataFrame into the 'predictions' table.

        Args:
            df (pd.DataFrame): The DataFrame containing predictions to insert.
        """

        self.insert_data(df, "predictions")


    def update_train_data(self, df):
        """
        Updates the 'train_data' table with new data (full replacement in this example).

        Args:
            df (pd.DataFrame): The DataFrame containing new training data.
        """
        self.connect_to_db()  # Ensure connection is established

        if not self.conn:
            raise ConnectionError("Failed to connect to database.")

        try:
            cursor = self.conn.cursor()
            # Delete all existing rows from 'train_data' (full replacement)
            cursor.execute("DELETE FROM train_data")
            self.insert_data(df, "train_data")  # Insert new data
            self.conn.commit()
            print("Train data updated with new data.")
        except Exception as e:
            self.conn.rollback()  # Rollback on errors
            raise e
        finally:
            if self.conn:
                cursor.close()

    def forget_predictions(self):
        """Deletes all predictions from the 'predictions' table."""
        self.connect_to_db()  # Ensure connection is established

        if not self.conn:
            raise ConnectionError("Failed to connect to database.")

        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM predictions")
            self.conn.commit()
            print("All predictions deleted.")
        except Exception as e:
            self.conn.rollback()  # Rollback on errors
            raise e

        finally:
            if self.conn:
                cursor.close()

    def get_data(self, table_name):
        """Retrieves data from a table in the PostgreSQL database as a DataFrame.

        Args:
            table_name (str): The name of the table to retrieve data from.

        Returns:
            pd.DataFrame: The DataFrame containing the retrieved data.
        """
        self.connect_to_db()  # Ensure connection is established

        if not self.conn:
            raise ConnectionError("Failed to connect to database.")

        try:
            cursor = self.conn.cursor()
            sql = f"SELECT * FROM {table_name}"  # Adjust query for specific columns if needed
            cursor.execute(sql)
            data = cursor.fetchall()
            df = pd.DataFrame(data, columns=[col.name for col in cursor.description])  # Create DataFrame
            return df
        except Exception as e:
            raise e
        finally:
            if self.conn:
                cursor.close()

    def get_train_data(self):
        return self.get_data("train_data")

    def get_new_data(self):
        return self.get_data("new_data")

    def get_predictions(self):
        return self.get_data("predictions")

    def get_joined_data_for_training(self):
        df1 = self.get_train_data()
        df2 = self.get_new_data()

        if len(df1)>0 and len(df2)>0:
            df = pd.concat([df1, df2], ignore_index=True)
        elif len(df1)<=0:
            df = df2
        elif len(df2)<=0:
            df = df1
        else:
            raise Exception("No data found")
        return df

    def get_join_data_for_data_drift(self):
        df1 = self.get_predictions()
        df2 = self.get_new_data()

        if len(df1) > 0 and len(df2) > 0:
            df = pd.concat([df1, df2], ignore_index=True)
        elif len(df1) <= 0:
            df = df2
        elif len(df2) <= 0:
            df = df1
        else:
            raise Exception("No data found")
        return df.drop(['cid'])



