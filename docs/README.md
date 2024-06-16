# Printing .md docs to pdf
```bash
pandoc koncepcja.md -s -o koncepcja.pdf -V colorlinks=true
```

	-s - standalone
	-o - po prostu plik wyjściowy
	-V colorlinks=true powoduje że hyperlinki są niebieskie (inaczej są czarne i nie widać że to hyperlinki)

## możliwe problemy
### bullet list doesnt work - appears in one line
to jest złe:
```
## Opis problemu
Problem dotyczy wieloetykietowej klasyfikacji przewinień z meczów piłki nożnej. Dla każdej akcji widzianej z wielu perspektyw należy przypisać dwie etykiety:
- pierwsza etykieta określa, czy wystąpił faul wraz z odpowiadającym mu stopniem powagi:
    - *No Offence*
    - *Offence + No Card*
```
musi być odstęp przed listą:
```
## Opis problemu
Problem dotyczy wieloetykietowej klasyfikacji przewinień z meczów piłki nożnej. Dla każdej akcji widzianej z wielu perspektyw należy przypisać dwie etykiety:

- pierwsza etykieta określa, czy wystąpił faul wraz z odpowiadającym mu stopniem powagi:
    - *No Offence*
    - *Offence + No Card*
```
