#!/usr/bin/env python
"""
Custom exception module for the hotel-reservation-prediction project.

This module defines CustomException, which enriches error messages with
contextual information such as the source file, line number, and original
exception details for easier debugging.
"""
import sys
import traceback
from typing import Optional, Type


class CustomException(Exception):
    """
    An exception wrapper that adds detailed context to error messages.

    Attributes:
        message (str): Formatted error message including file and line context.
        original_exception (Optional[BaseException]): The underlying exception instance, if any.
    """

    def __init__(
        self,
        message: str,
        original_exception: Optional[BaseException] = None
    ):
        """
        Initialize CustomException with a custom message and optional original exception.

        Args:
            message (str): High-level description of the error.
            original_exception (Optional[BaseException]): The caught exception instance.
        """
        # Store the original exception
        self.original_exception = original_exception
        # Build and store the detailed message
        detailed = self._build_detailed_message(message, original_exception)
        super().__init__(detailed)
        self.message = detailed

    @staticmethod
    def _build_detailed_message(
        message: str,
        original_exception: Optional[BaseException]
    ) -> str:
        """
        Construct a detailed error message with file name, line number, and original exception.

        Args:
            message (str): High-level description of the error.
            original_exception (Optional[BaseException]): The caught exception instance.

        Returns:
            str: Formatted message with context.
        """
        # Determine traceback: prefer original exception's traceback if provided
        tb = (
            original_exception.__traceback__
            if original_exception is not None
            else sys.exc_info()[2]
        )

        if tb:
            frame = tb.tb_frame
            lineno = tb.tb_lineno
            filename = frame.f_code.co_filename
            base = f"Error in '{filename}', line {lineno}: {message}"
        else:
            base = message

        if original_exception:
            return f"{base} | Original exception: {original_exception!r}"
        return base

    def __str__(self) -> str:
        return self.message
