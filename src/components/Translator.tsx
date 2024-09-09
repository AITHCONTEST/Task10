import { observer } from "mobx-react-lite";
import LangButt from "./LangButt";
import langStore from "../store/langStore";
import styles from "./styles/translator.module.scss"
import { useEffect, useRef, useState } from "react";
import translateService from "../service/translateService";
import Loader from "./Loader";
import CopyButt from "./CopyButt";
import KeyboardButt from "./KeyboardButt";
import langService from "../service/langService";
import ClearButt from "./ClearButt";

const Translator = observer(() => {
    const [input, setInput] = useState<string>("");
    const [output, setOutput] = useState<string>("");
    const [loading, setLoading] = useState<boolean>(false);
    const maxLength = 5000; 
    const inputAreaRef = useRef<HTMLTextAreaElement>(null);
    const outputAreaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(()=>{
        fixHeights();
    }, [input, output])

    useEffect(()=>{
        const translate = async () => {
            setLoading(true);
            const fromLang = langStore.fromLang;
            const toLang = langStore.toLang;
            const result = await translateService.translate(input, fromLang, toLang);
            setOutput(result);
            setLoading(false);
        }

        if(input.length){
            translate();
        }
        else{
            setOutput("")
        }
        
    }, [input, output, langStore.fromLang])

    


    const fixHeights = () => {
        if (inputAreaRef.current && outputAreaRef.current) {
            inputAreaRef.current.style.height = 'auto'; 
            outputAreaRef.current.style.height = 'auto'; 
            const height = Math.max(inputAreaRef.current.scrollHeight, outputAreaRef.current.scrollHeight)
            inputAreaRef.current.style.height = `${height}px`; 
            outputAreaRef.current.style.height = `${height}px`;
        }
    }

    return(
        <div className={styles.translator} >
            <div className={styles.switcher}>
                <div className={styles.switcher__lang}>{langService.getNameByCode(langStore.fromLang)}</div>
                <LangButt/>
                <div className={styles.switcher__lang}>{langService.getNameByCode(langStore.toLang)}</div>
            </div>
            <div className={styles.container}>
                <div className={styles.field}>
                    <hr className={styles.field__hr}/>
                    <div className={styles.field__wrapper} style={{marginLeft:"auto"}}>
                        <textarea
                            placeholder="Введите текст для перевода..."
                            value={input}
                            ref={inputAreaRef}
                            className={styles.field__text}
                            maxLength={maxLength}
                            onChange={e => setInput(e.currentTarget.value || "")}
                        >
                           
                        </textarea>
                        <div className={styles.field__tools} >
                            <div>{maxLength-input.length}/{maxLength}</div>
                            <div className={styles.field__tools_wrap}>
                                <KeyboardButt get={()=>input} set={(newInput:string)=>{setInput(newInput)}} />
                                <ClearButt  tap={()=>{setInput("")}}/>
                            </div>
                            
                            
                        </div>
                    </div>
                </div>
                <div className={styles.field}>
                    <hr className={styles.field__hr} />
                    <div className={styles.field__wrapper} style={{marginRight:"auto"}}>
                        <textarea
                            value={output}
                            ref={outputAreaRef}
                            className={`${styles.field__text} ${styles.field__text_readOnly}`}
                            readOnly
                        >
                        </textarea>
                        {loading && <Loader/>}
                        <div className={styles.field__tools} >
                            <div className={styles.field__tools_wrap}>
                                <CopyButt targetRef={outputAreaRef}/>
                            </div>

                            <div className={styles.field__tools_wrap}>
                                
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            
        </div>
    );
});

export default Translator;