"""
Phase 1: Hello World Tracing

Goal: Understand how MLflow tracing works with a simple function.
Run this first to see how traces capture function calls.

To run:
    python examples/01_hello_trace.py

Then view in MLflow UI:
    mlflow ui
    # Open http://localhost:5000
"""

import mlflow


@mlflow.trace
def greet(name: str) -> str:
    """A simple function decorated with @mlflow.trace"""
    greeting = f"Hello, {name}!"
    return greeting


@mlflow.trace
def greet_with_processing(name: str) -> dict:
    """A function that calls another traced function"""
    processed_name = name.strip().title()
    message = greet(processed_name)

    return {
        "original": name,
        "processed": processed_name,
        "message": message
    }


if __name__ == "__main__":
    # Simple trace
    result1 = greet("alice")
    print(f"Result 1: {result1}")

    # Nested trace
    result2 = greet_with_processing("  bob smith  ")
    print(f"Result 2: {result2}")

    print("\nTraces created! View them in MLflow UI:")
    print("   Run: mlflow ui")
    print("   Open: http://localhost:5000")
    print("\nLook for:")
    print("   - Trace timeline showing function calls")
    print("   - Input/output parameters")
    print("   - Nested spans (greet_with_processing calls greet)")
