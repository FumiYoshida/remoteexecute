# 標準ライブラリ
import os
import datetime
import base64
import pickle
import types
import threading
import traceback
import warnings
import logging
import inspect
import functools
from pathlib import Path

# PyPI
import requests
import pandas as pd

# PyPI - Flask関連
from flask import Flask, request, jsonify
from werkzeug.serving import make_server

DATA_TO_LOG_KEY = '_remoteexecute_data_to_log' # 関数の実行には使わないがログには保存するデータを入れる引数名

def obj_to_base64(obj):
    """
    Serialize an object and convert it to a base64 encoded string.

    Parameters
    ----------
    obj : object
        The object to be serialized and encoded.

    Returns
    -------
    str
        A base64 encoded string that represents the serialized object.
    """
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

def base64_to_obj(base64_data):
    """
    Convert a base64 encoded string to a python object by decoding and deserializing it.

    Parameters
    ----------
    base64_data : str or bytes
        The base64 encoded representation of the object.

    Returns
    -------
    object
        The deserialized python object.
    """
    if isinstance(base64_data, bytes):
        return pickle.loads(base64.b64decode(base64_data))
    else:
        return pickle.loads(base64.b64decode(base64_data.encode('utf-8')))

class SerializedArgs:
    """
    A class for serializing and deserializing arguments of a function.
    """
    @staticmethod
    def serialize(func, args, kwargs):
        """
        Serialize the function's details and its arguments.

        Parameters
        ----------
        func : function
            The function whose details are to be serialized.
        args : tuple
            The positional arguments of the function.
        kwargs : dict
            The keyword arguments of the function.

        Returns
        -------
        str
            A base64 encoded string containing the serialized data.
        """
        request_data = {
            'pid': os.getpid(), # 60ns程度
            'func_name': func.__name__,
            'args': args,
            'kwargs': kwargs,
        }
        
        return obj_to_base64(request_data)
    
    @staticmethod
    def deserialize(parent, res):
        """
        Deserialize the function's details and its arguments.

        Parameters
        ----------
        parent : object
            An object that has the function as its method.
        res : str
            A base64 encoded string containing the serialized data.

        Returns
        -------
        dict
            A dictionary containing the deserialized function and its arguments.
        """
        # データを復元
        request_data = base64_to_obj(res)

        # 関数名 -> 関数object に変換
        request_data['func'] = getattr(parent, request_data.pop('func_name'))
        
        return request_data

class SerializedResult:
    """
    A class for serializing and deserializing the results of function execution.
    """
    @staticmethod
    def call_and_serialize(pid, func, args, kwargs):
        """
        Execute a function and serialize its result or exception if any.

        Parameters
        ----------
        pid : int
            The process id of the client.
        func : function
            The function to be executed.
        args : tuple
            The positional arguments of the function.
        kwargs : dict
            The keyword arguments of the function.

        Returns
        -------
        str
            A base64 encoded string containing the serialized result or exception details.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Capture all warnings
            
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                # 例外とトレースバックを取得
                error_message = str(e)
                tb = traceback.format_exc()

                # response_dataにエラーメッセージとトレースバックを格納
                response_data = {
                    'error': error_message,
                    'traceback': tb,
                }
            else:
                response_data = {
                    'result': obj_to_base64(result),
                }
                
            # Catch warnings
            if w:
                warnings_list = [str(warning.message) for warning in w]
                response_data['warnings'] = warnings_list
                
        return obj_to_base64(response_data)
        
    @staticmethod
    def deserialize(res):
        """
        Deserialize the result of function execution or raise exception if any.

        Parameters
        ----------
        res : str
            A base64 encoded string containing the serialized result or exception details.

        Returns
        -------
        object
            The deserialized result of the function execution.

        Raises
        ------
        Exception
            If there was an exception during function execution, it is raised again.
        """
        response_data = base64_to_obj(res)
        
        # If there are warnings in the response, output them
        if 'warnings' in response_data:
            for warning_msg in response_data['warnings']:
                warnings.warn(warning_msg)

        if 'error' in response_data and 'traceback' in response_data:
            # サーバからのエラーとトレースバックを取得
            error_message = response_data['error']
            tb = response_data['traceback']
            
            # エラーを再現
            raise Exception(f"Server Error: {error_message}\nServer Traceback: {tb}")
        else:
            result = base64_to_obj(response_data['result'])
            return result
        
def remote_method_decorator(method, server_url, instance, log_class_attrs):
    """
    Decorator to make a method execute remotely.

    Parameters
    ----------
    method : function
        The method to be decorated.
    server_url : str
        The URL where the server is hosted.
    instance : object
        The instance of the class where the method is defined. This instance 
        is used to retrieve the attribute values specified in `log_class_attrs`.
    log_class_attrs : list of str
        A list of attribute names whose values should be logged and sent with 
        the request. These attributes are retrieved from the provided instance.
    
    Returns
    -------
    function
        The decorated function that executes the method remotely.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # ログに追加すべきクライアント側の情報をまとめる
        kwargs[DATA_TO_LOG_KEY] = {}
        for attr_name in log_class_attrs:
            kwargs[DATA_TO_LOG_KEY][attr_name] = getattr(instance, attr_name, None)
        
        # 実行に必要な情報をstring型にまとめる
        request_data = SerializedArgs.serialize(method, args, kwargs)

        # リクエストを送る
        response = requests.post(server_url, data=request_data)
        
        # 実行結果を復元して返す
        result = SerializedResult.deserialize(response.text)
        return result
    
    return wrapper

def create_server_client_classes(original_class, host="localhost", port="5000", 
                                 visible_from_outside=False, is_server_func=None, 
                                 log_execution_info=True, log_dir='./log/', log_exclude_funcs=None,
                                 log_class_attrs=None,):
    """
    Create Server and Client classes based on the given class for remote method execution.

    Parameters
    ----------
    original_class : class
        The base class to create the Server and Client classes from.
    host : str, optional
        The host where the server will be run.
    port : str, optional
        The port where the server will be listening.
    visible_from_outside : bool, default: False
        Whether the Server Class is accessible from outside the PC where it is being run.
        Even if this is set to True, access is not possible unless port forwarding is done with an ssh connection.
    is_server_func : callable, optional
        A function that takes the name of a function as input
        and returns a bool indicating whether it should be a server-side only function
        (i.e., a function that cannot be executed by the client).
    log_execution_info : bool, default: True
        Whether to record execution info such as called time, function name, serialized input, serialized output.
    log_dir : str, optional
        directory to save log file.
    log_exclude_funcs: list of str, optional
        list of function names to exclude from the record.
        
    Returns
    -------
    tuple
        A tuple containing the Server and Client classes.
    """
    # URLの設定
    
    #"localhost"だとipv6(::1)にアクセスすることを試み、失敗後ipv4(127.0.0.1)で接続するため、
    # flaskで"localhost"を指定すると2sほどの遅延が生じる
    # そのため、直接IPアドレスを指定する
    if host == "localhost":
        host = "127.0.0.1"
    
    execute_url_suffix = "/execute"
    init_arg_url_suffix = "/init_args"
    server_url = f"http://{host}:{port}{execute_url_suffix}"
    init_arg_url = f"http://{host}:{port}{init_arg_url_suffix}"
            
    # リクエストのたびに出力が出ないようにする
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    is_dunder_func =  lambda func_name: func_name.startswith("__") and func_name.endswith("__")
    if is_server_func is None:
        is_server_func = lambda func_name: False

    if log_exclude_funcs is None:
        log_exclude_funcs = []

    # 各種ログの追加設定
    if log_class_attrs is None:
        log_class_attrs = []
    
    class Server(original_class):
        @functools.wraps(original_class.__init__)
        def __init__(self, *args, **kwargs):
            # 元クラスの初期化
            super().__init__(*args, **kwargs)
            
            # 初期化の引数の情報を記録
            bound_args = inspect.signature(super().__init__).bind(*args, **kwargs)
            bound_args.apply_defaults()
            args_dict = dict(bound_args.arguments)
            
            # Flaskアプリを設定
            self._app = Flask(__name__)
            self._is_visible_from_outside = visible_from_outside
            self._host = '0.0.0.0' if self._is_visible_from_outside else host
            self._port = port

            # postを受け取って関数を実行するよう設定
            @self._app.route(execute_url_suffix, methods=['POST'])
            def execute():
                serialized_args = request.get_data()
                execute_info = SerializedArgs.deserialize(self, serialized_args) # flask.request
                
                func_name = execute_info['func'].__name__
                if is_server_func(func_name) or is_dunder_func(func_name):
                    # 無効な関数名を指定されたら
                    # 実行せずに返す
                    return "nice try!"

                if DATA_TO_LOG_KEY in execute_info['kwargs']:
                    data_to_log = execute_info['kwargs'].pop(DATA_TO_LOG_KEY)
                else:
                    data_to_log = {}
                
                serialized_result = SerializedResult.call_and_serialize(**execute_info)

                if log_execution_info and (func_name not in log_exclude_funcs):
                    # 記録対象の関数のとき 実行情報を保存
                    now = datetime.datetime.now()
                    self._check_date_change(now)
                    print({
                        'time': str(now),
                        'args': serialized_args,
                        'result': serialized_result,
                    }, file=self._log_file, flush=True)
                
                return serialized_result
            
            # __init__の引数の情報をリクエストされたら返すよう設定
            @self._app.route(init_arg_url_suffix, methods=['GET'])
            def init_arg():
                return obj_to_base64(args_dict)
            
            # サーバーを起動
            self.start_server()

        def _check_date_change(self, now):
            if now.date() != self._log_file_date:
                # 実行中に日付が変わったら
                # 保存先のファイルを変更する
                self._log_file.close()
                self._log_file = open(Path(log_dir) / f"{now.date()}.txt", mode='a', encoding='utf-8-sig')
        
        def start_server(self):
            """サーバーを起動する"""
            if log_execution_info:
                self._log_file_date = datetime.date.today()
                Path(log_dir).mkdir(exist_ok=True)
                self._log_file = open(Path(log_dir) / f"{datetime.date.today()}.txt", mode='a', encoding='utf-8-sig')
            self._server = make_server(self._host, self._port, self._app)
            self._thread = threading.Thread(target=self._server.serve_forever)
            self._thread.start()

        def stop_server(self):
            """サーバーを停止する　以降`start_server`を呼ぶまでrequestを受け付けなくなる"""
            self._server.shutdown()
            self._thread.join()
            if log_execution_info:
                self._log_file.close()

            
    class Client(original_class):
        @functools.wraps(original_class.__init__)
        def __init__(self, *args, **kwargs):
            # 初期化処理はサーバー側で行うのでこれは実行しない
            #super().__init__(*args, **kwargs)
            
            # 初期化の引数の情報を記録
            bound_args = inspect.signature(super().__init__).bind(*args, **kwargs)
            bound_args.apply_defaults()
            client_args_dict = dict(bound_args.arguments)
            
            # サーバー側の初期化時の引数の情報を確認
            res = requests.get(init_arg_url)
            server_args_dict = base64_to_obj(res.text)
            
            # サーバー側とクライアント側で引数が一致しているか確認
            # 一致していなければサーバー側の値が優先される旨をwarning
            assert set(client_args_dict) == set(server_args_dict), "サーバーとクライアントのバージョンが異なります"
            for key in client_args_dict.keys():
                client_value = client_args_dict[key]
                server_value = server_args_dict[key]
                if client_value != server_value:
                    warning_str = (
                        f"引数`{key}`の設定がサーバー({server_value})とクライアント({client_value})で異なります。"
                        f"サーバー側の値({server_value})が優先されます。"
                    )
                    warnings.warn(warning_str)
            
            def server_only_func(*args, **kwargs):
                warnings.warn('server-only func')
            
            # すべての関数について、実行する代わりにサーバーに実行をリクエストするよう変更
            for attr_name in dir(original_class):  # dirを使用してすべてのメンバを取得
                attr_value = getattr(original_class, attr_name)
                # オブジェクトのメソッドのみデコレート（dunderメソッドを除く）
                if callable(attr_value):
                    if is_dunder_func(attr_name):
                        pass
                    elif is_server_func(attr_name):
                        setattr(self, attr_name, server_only_func)
                    else:
                        decorated_method = remote_method_decorator(
                            method=attr_value,
                            server_url=server_url,
                            instance=self,
                            log_class_attrs=log_class_attrs,
                        )
                        setattr(self, attr_name, types.MethodType(decorated_method, self))

    return Server, Client

def read_log_file(path, reconstruct=True):
    # ログファイルを読み込む
    with open(path, encoding='utf-8-sig', mode='r') as f:
        log_df = pd.DataFrame(map(eval, f.readlines()), columns=['time', 'args', 'result'])
        log_df['time'] = pd.to_datetime(log_df['time'])
        log_df = log_df.fillna('')
    
    if reconstruct and len(log_df) > 0:
        # オブジェクトを再構成するとき

        # 引数と追加ログを再構成
        args_objs = []
        for serialized_args in log_df['args']:
            args_obj = base64_to_obj(serialized_args)
            if DATA_TO_LOG_KEY in args_obj['kwargs']:
                args_obj['additional_log'] = args_obj['kwargs'].pop(DATA_TO_LOG_KEY)
            else:
                args_obj['additional_log'] = {}
            args_objs.append(args_obj)
        args = pd.DataFrame(args_objs)

        # 戻り値を再構成
        results = pd.DataFrame(list(log_df['result'].map(base64_to_obj)))
        results['result'] = results['result'].map(base64_to_obj, na_action='ignore')
        results.fillna('', inplace=True)

        # まとめてpandas.DataFrameにして返す
        reconstructed_log_df = pd.concat([log_df[['time']], args, results], axis=1)
        return reconstructed_log_df
    else:
        return log_df

def read_log_dir(dir, reconstruct=True):
    dir = Path(dir)
    log_dfs = []
    for path in dir.glob('*.txt'):
        df = read_log_file(path)
        if len(df) > 0:
            log_dfs.append(df)
    
    return pd.concat(log_dfs).reset_index(drop=True)