const AI_API_URL = 
    import.meta.env.VITE_ISLOCAL ? 
        'http://localhost:5000/api'
    :
        'https://mohamed11ismail.pythonanywhere.com/api';

export { AI_API_URL };


