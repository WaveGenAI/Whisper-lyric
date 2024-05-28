import os
import random
import re
import typing

import orjson
import requests as r

_BASE_URL = (
    "https://firestore.googleapis.com/google.firestore.v1.Firestore/Listen/channel"
)
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "accept": "*/*",
    "accept-language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "content-type": "application/x-www-form-urlencoded",
    "priority": "u=1, i",
    "sec-ch-ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "cross-site",
}


class SonautoAPI:
    """
    Class to download the audio from sonauto for finetuning the whisper model
    """

    def __init__(self, clean: bool = False):
        self.session = r.Session()
        self._zx = "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz1234567890") for i in range(10)
        )

        self._aid = 0
        self._limit_results = 25

        self._init_dir()

        if clean:
            self._clean_dir()

    def _get_url(self, gsession_id: str = None, sess_id: str = None) -> str:
        """
        Method to get the url to download the data

        Returns:
            str: The url to download the data
        """

        url = (
            f"{_BASE_URL}?VER=8&database=projects%2Fsonauto-abefa%2Fdatabases%2F(default)&"
            f"RID=95901&CVER=22&X-HTTP-Session-Id=gsessionid&zx={self._zx}&t=1"
        )
        if gsession_id and sess_id:
            url = (
                f"{_BASE_URL}?gsessionid={gsession_id}&VER=8&database=projects%2Fsonauto-abefa%2Fdatabases%2F(default)&"
                f"RID=rpc&SID={sess_id}&AID={self._aid}&CI=1&TYPE=xmlhttp&zx={self._zx}&t=1"
            )

            self._aid += 1
        return url

    def _get_session_id(self, nb_ignored_val: int = 0) -> typing.Tuple[str, str]:
        """Method to get the session id

        Args:
            nb_ignored_val (int, optional): the number of data to ignore before running the request. Defaults to 0.

        Raises:
            r.exceptions.HTTPError: If the request failed

        Returns:
            typing.Tuple[str, str]: The session id and the gsession id
        """

        req = (
            "headers=X-Goog-Api-Client:gl-js/ fire/9.23.0\n"
            "Content-Type:text/plain\n"
            "X-Firebase-GMPID:1:134904337177:web:e59fdf0aec84eb16ceeb88\n"
            "&count=1&ofs=0&req0___data__="
        ).strip()

        body = {
            "database": "projects/sonauto-abefa/databases/(default)",
            "addTarget": {
                "query": {
                    "structuredQuery": {
                        "from": [{"collectionId": "songs"}],
                        "orderBy": [
                            {
                                "field": {"fieldPath": "projectId"},
                                "direction": "ASCENDING",
                            },
                        ],
                        "limit": self._limit_results,
                        "offset": nb_ignored_val,
                    },
                    "parent": "projects/sonauto-abefa/databases/(default)/documents",
                },
                "targetId": 1,
            },
        }

        req += orjson.dumps(body).decode("utf-8")
        response = self.session.post(self._get_url(), headers=_HEADERS, data=req)

        if response.status_code != 200:
            raise r.exceptions.HTTPError(
                f"Failed to get session id, status code: {response.status_code}"
            )

        sess_id = re.search(r'c","([^"]+)', response.text).group(1)
        gsession_id = response.headers.get("X-HTTP-Session-Id")

        return sess_id, gsession_id

    def _init_dir(self) -> None:
        """Method to init the directory"""

        if not os.path.exists("./dataset/lyrics"):
            os.makedirs("./dataset/lyrics")

        if not os.path.exists("./dataset/audio"):
            os.makedirs("./dataset/audio")

    def _clean_dir(self) -> None:
        """Method to clean the directory"""

        for file in os.listdir("./dataset/lyrics"):
            os.remove(f"./dataset/lyrics/{file}")

        for file in os.listdir("./dataset/audio"):
            os.remove(f"./dataset/audio/{file}")

    def _construct_ds(self, audio_data: dict) -> None:
        """Method to construct the dataset

        Args:
            audio_data (dict): the audio data
        """

        for _, value in audio_data.items():
            nbm_file = len(
                [i for i in os.listdir("./dataset/lyrics") if i.endswith(".txt")]
            )

            with open(f"./dataset/lyrics/{nbm_file}.txt", "w") as file:
                file.write(value["lyrics"])

            with self.session.get(
                "https://cdn.sonauto.ai/generations/" + value["audio_url"]
            ) as resp:
                with open(f"./dataset/audio/{nbm_file}.ogg", "wb") as file:
                    file.write(resp.content)

            print(f"Downloaded {nbm_file} songs.", end="\r")

    def download(self, nb_element: int = 10) -> None:
        """Method to download the data

        Args:
            nb_element (int, optional): the number of item to download. Defaults to 10.
        """

        nbm_dl_song = 0
        out = {}

        while nbm_dl_song < nb_element:
            sess_id, gsession_id = self._get_session_id(len(out.keys()))

            with self.session.get(
                self._get_url(gsession_id, sess_id), headers=_HEADERS
            ) as resp:
                size = int(resp.text.split("\n")[0])
                if size < 1000:  # too small (finishing the loop)
                    continue

                content = "\n".join(resp.text.split("\n")[1:])[:size]
                if re.search(r"\]\d{2,}(\\n|\n)", content):
                    content = re.sub(r"\]\d{2,}(\\n|\n)", "", content)
                    content += "]"
                json_data = orjson.loads(content)

                for element in json_data:
                    if "documentChange" not in element[1][0]:
                        continue

                    element = element[1][0]["documentChange"]["document"]
                    out[str(len(out.keys()))] = {
                        "lyrics": element["fields"]["lyrics"]["stringValue"],
                        "audio_url": element["fields"]["audioPath"]["stringValue"],
                        "prompt": element["fields"]["prompt"]["stringValue"],
                    }
                    nbm_dl_song += 1
                    if nbm_dl_song >= nb_element:
                        break

            self._construct_ds(out)
            out = {}
