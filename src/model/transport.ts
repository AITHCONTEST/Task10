
interface ITransport{
    post(text:string, language:string):Promise<string>;
}

export default ITransport;