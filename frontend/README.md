Для запуска склонируйте репозиторий (понадобиться node.js и npm)
и выполните следующие команды

- npm i

- npm run build   

- npm run dev         


или 

- docker build -t translate2win_front .      
- docker run -p 3000:3000 translate2win_front


в файле src/transport/Transport.ts
указать внешний адрес сервера на котором забущен проект/backend 
