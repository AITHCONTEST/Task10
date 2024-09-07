import { observer } from "mobx-react-lite";
import langService from "../service/langService";
import toggle from "../assets/toggle.svg";
import Button from "./Button";

const LangButt = observer(() => {
    const tap = ()=>{
        langService.toggleLang();
    }    
    return(
        <>
            <Button img={toggle} tap={tap}  alt={"<->"}/>
        </>
    );
});

export default LangButt;