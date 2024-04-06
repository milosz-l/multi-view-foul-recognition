# Temat
Soccernet - multi-view foul recognition - https://www.soccer-net.org/tasks/new-multi-view-foul-recognition

## Zespół
- Adam Górski, 304054
- Miłosz Łopatto, ...

## Opis rozwiązania zadania

#### Opis zadania
Konkurs polega na wytrenowaniu jak najlepszego rozwiązania typu VAR, które na podstawie krótkich klipów potrafi klasyfikować przewinienia podczas meczów piłki nożnej. Model ma jednocześnie radzić sobie jak najlepiej z wieloma zadaniami i potrafić sobie radzić ze zróżnicowaną liczbą klipów źródłowych. Poza zbiorem treningowym dostępne są jeszcze zbiór testowy, na podstawie którego zespoły są oceniane podczas pierwszego etapu i ukryty, którego wyniki będą oceniane podczas ostatniego etapu.

#### Opis rozwiązania
Jest to model klasyfikujący wideo pochodzące z wielu klipów i uczący się jednocześnie wielu zadań - czy jest to faul, jaki jest to typ faulu oraz jak bardzo poważne jest to przewinienie.
Pierwszą częścią architektury jest model wyciągający cechy z klipów wideo. Domyślnie wykorzystywane są wcześniej wytrenowane modele wideo z biblioteki torchvision - takie jak r3d_18, s3d, mc3_18, r2plus1d_18 i mvit_v2_s. Kolejną warstwą jest aggregator, który agreguje wyniki z kilku ekstraktorów. Ostatecznym elementem jest głowica do klasyfikacji wielozadaniowej. Poza próbą jak najlepszego dostrojenia warstwy agregującej planujemy także eksperymentować z zastąpieniem jej modelem opartym o atencję.

#### Zbiór danych
Zbiór danych pochodzi z konkursu Soccernet - zawiera ponad 2000 oanotowanych akcji, gdzie do każdej akcji jest od 2 do 5 klipów. Klipy są długości do 164 klatek i mają 24 klatki na sekundę. Zbiory testowe i ukryty mają po 250 klipów.

#### Narzędzia
Rozwiązanie będzie zaimplementowane w Pythonie w wersji 3.11 z użyciem biblioteki torch, torchvision i pytorch lightning. Dodatkowo wyniki
eksperymentów będą śledzone przy pomocy weights and biases.

