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
    def __init__(self):
        self.session = r.Session()
        self._zx = "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz1234567890") for i in range(10)
        )

        self._aid = 0
        self._limit_results = 25

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
                        "where": {
                            "fieldFilter": {
                                "field": {"fieldPath": "public"},
                                "op": "EQUAL",
                                "value": {"booleanValue": True},
                            }
                        },
                        "orderBy": [
                            {
                                "field": {"fieldPath": "createdAt"},
                                "direction": "DESCENDING",
                            },
                            {
                                "field": {"fieldPath": "__name__"},
                                "direction": "DESCENDING",
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

    def download(self, nb_element: int = 10) -> dict:
        """Method to download the data

        Args:
            nb_element (int, optional): the number of item to download. Defaults to 10.

        Returns:
            dict: The data
        """

        out = {}

        while len(out.keys()) < nb_element:
            sess_id, gsession_id = self._get_session_id(len(out.keys()))

            with self.session.get(
                self._get_url(gsession_id, sess_id), headers=_HEADERS
            ) as resp:
                size = int(resp.text.split("\n")[0])
                content = "\n".join(resp.text.split("\n")[1:])[:size]

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

                    if len(out.keys()) >= nb_element:
                        break

        return out
