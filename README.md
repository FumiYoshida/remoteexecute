
# RemoteExecute
This Python utility allows for remote method execution over HTTP. It encompasses a range of functionalities such as object serialization and deserialization, and remote method invocation via HTTP POST requests.
<p align="center">
 <img src="https://img.shields.io/badge/python-v3.9+-blue.svg">
 <img src="https://img.shields.io/badge/contributions-welcome-orange.svg">
 <a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg">
 </a>
</p>

## Install Requirements
```bash
pip install git+https://github.com/FumiYoshida/remoteexecute
```

## Sample Code

Here is a simple example demonstrating how to use the `create_server_and_client_classes` function to execute methods remotely.

### 1. Define a Class with Methods to Execute Remotely

```python
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
```

### 2. Create Server and Client Classes

```python
from your_module import create_server_and_client_classes

ServerClass, ClientClass = create_server_and_client_classes(Calculator)
```

### 3. Start the Server

```python
server = ServerClass()
```

### 4. Make Remote Method Calls from the Client

```python
client = ClientClass()

result = client.add(2, 3)
print(f"2 + 3 = {result}")

result = client.subtract(5, 1)
print(f"5 - 1 = {result}")
```

### 5. Stop the Server

```python
server.stop_server()
```

In this example, a `Calculator` class is defined with `add` and `subtract` methods. Server and Client classes are created from this class, and then the server is started. The client can then make remote method calls, and the results will be printed out. Finally, the server is stopped.