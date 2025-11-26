**Krystalografia-Kalkulator**

Prosty interaktywny skrypt wizualizujący płaszczyzny sieciowe w sześcianie przy pomocy `matplotlib`.

**Uruchamianie w Visual Studio Code**

**Najprostsze (Mac / Windows) — bez `venv`**

Mac (Terminal / zsh):

```bash
# Przejdź do folderu projektu
cd /ścieżka/do/krystalografia-kalkulator

# Zainstaluj wymagane biblioteki (jednokrotne)
python3 -m pip install -r requirements.txt

# Uruchom program
python3 app.py
```

Windows (Wiersz poleceń - Command Prompt):

```cmd
:: Przejdź do folderu projektu
cd C:\sciezka\do\krystalografia-kalkulator

:: Zainstaluj wymagane biblioteki (jednokrotne)
python -m pip install -r requirements.txt

:: Uruchom program
python app.py
```

Krótko:
- Najpierw instalujesz zależności poleceniem `pip install -r requirements.txt`.
- Potem uruchamiasz `app.py` przy pomocy `python app.py` (macOS: `python3 app.py`).
- Jeśli coś nie działa, skopiuj i wklej komunikat błędu do wyszukiwarki lub daj znać — pomogę.

1. Otwórz folder projektu w VS Code (`File -> Open Folder...`).
2. Zainstaluj rozszerzenie `Python` (Microsoft).
3. Wybierz interpreter Pythona: kliknij w prawym dolnym rogu na nazwę interpretera i wybierz `./.venv/bin/python` (jeśli używasz venv) lub systemowy interpreter.
4. Zainstaluj wymagane pakiety (z terminala w VS Code):

```bash
python -m pip install -r requirements.txt
```

5. Uruchom plik `app.py` bezpośrednio z edytora (prawy przycisk -> `Run Python File in Terminal`) albo użyj Debuggera.

