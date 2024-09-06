

interface ITransport{
    post(text:string, from:string, to:string):Promise<string>;
}

export default ITransport;