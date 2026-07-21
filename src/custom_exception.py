# This file creates a custom exception class to show better error messages.
import traceback
import sys


class CustomException(Exception):
    """A custom exception that shows where the error happened."""

    def __init__(self, error_message, error_detail: sys):
        # Call the parent Exception class and keep the main error message.
        super().__init__(error_message)

        # Build a detailed message with the file name and line number.
        self.error_message = self.get_detailed_error_messages(error_message, error_detail)

    @staticmethod
    def get_detailed_error_messages(error_message, error_detail: sys):
        # Get the exception information from the error details.
        _, exc_value, exc_tb = traceback.sys.exc_info()

        # Find the file where the error happened.
        file_name = exc_tb.tb_frame.f_code.co_filename

        # Find the exact line number where the error happened.
        line_number = exc_tb.tb_lineno

        # Return a clearer error message, including the original exception's message.
        return f"Error in {file_name}, line {line_number}: {error_message} | Original error: {exc_value}"

    def __str__(self):
        # This makes the exception print the detailed message nicely.
        return self.error_message       